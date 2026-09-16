# renderer.py - Display, formatting, LaTeX rendering, and image handling
#
# See start.py for usage.

import os
import re
import base64
import shlex
import shutil
from pathlib import Path
from pylatexenc.latex2text import LatexNodes2Text
from rich.markdown import Markdown
from rich.syntax import Syntax
import importlib.util

import config
from utils import MIME_MAP, auto_enhance, console

# ── LaTeX converter ──────────────────────────────────────────────────────────
_latex_converter = LatexNodes2Text()

# ── LaTeX pre-processing, BEFORE pylatexenc ever sees the string ──────────────
# pylatexenc's LatexNodes2Text silently DISCARDS macros it doesn't recognize
# (rather than leaving them as literal text) — and since this runs before the
# _bare fallback dict below, anything dropped here never gets a second chance.
# \implies and \| are common cases that fall into this gap: \Rightarrow (which
# IS in pylatexenc's table) renders fine, while \implies vanishes with zero
# trace; a plain "|x|" survives untouched and later gets swapped to ∣, while
# the escaped "\|x\|" form is silently eaten. Mapping these to pylatexenc-safe
# text BEFORE conversion closes that gap.
_PRE_LATEX_MACRO_FIXUPS = {
    r'\implies':   ' ⇒ ',
    r'\iff':       ' ⇔ ',
    r'\therefore': ' ∴ ',
    r'\because':   ' ∵ ',
    r'\lVert':     '‖',
    r'\rVert':     '‖',
    r'\Vert':      '‖',
    r'\lvert':     '|',
    r'\rvert':     '|',
    # \| itself is ambiguous in real LaTeX (renders as ‖), but in this
    # codebase's usage it's consistently used as an escaped absolute-value
    # bar (\|x\|) alongside plain |x| in the SAME response — mapping it to a
    # plain pipe keeps it consistent with the existing '|' → '∣' swap below,
    # rather than silently disappearing.
    r'\|':         '|',
    r'\bmod':      ' mod ',
    r'\mod':       ' mod ',
    r'\degree':    '°',
    r'^\circ':     '°',   # \degree alias: 30^\circ → 30° (must be in PRE dict so it
                            # runs before pylatexenc converts \circ to ∘)
}


# ── Digit maps for compound fraction exponents ───────────────────────────────
# Module-level so _apply_pre_latex_fixups (math-block path) and the bare
# processing path in prep_for_console can both use them without import cycles.
_SUP_DIGIT = str.maketrans('0123456789', '\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079')
_SUB_DIGIT = str.maketrans('0123456789', '\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089')


def _fmt_frac_exp(m: re.Match) -> str:
    """Convert ^{3/2} to the form 3-over-2 using Unicode superscript/subscript digits
    separated by U+2044 FRACTION SLASH (\u2044).

    pylatexenc only converts the FIRST token of a compound exponent like ^{3/2},
    leaving the rest as plain text — producing the ambiguous "Δ3/2" which reads
    as "Δ-cubed divided by 2" instead of "Δ to the power 3/2". Calling this
    BEFORE pylatexenc avoids that mangling entirely.

    Mathematical correctness is preserved; the representation is slightly less
    pretty than ideal but is unambiguous and correct.
    """
    num = m.group(1).translate(_SUP_DIGIT)
    den = m.group(2).translate(_SUB_DIGIT)
    return f'{num}\u2044{den}'   # U+2044 FRACTION SLASH \u2044


def _apply_pre_latex_fixups(latex_src: str) -> str:
    """Apply pre-processing fixups to LaTeX source BEFORE pylatexenc sees it.

    pylatexenc silently drops macros it does not recognise. This function
    converts the most common dropped macros to safe equivalents so they
    survive the pylatexenc pass.
    """
    # ── Modular arithmetic, binomials, boxed elements ——————————————
    latex_src = re.sub(r'\\pmod\{([^{}]*)\}', r' (mod \1)', latex_src)
    latex_src = re.sub(r'\\pod\{([^{}]*)\}', r' (\1)', latex_src)
    latex_src = re.sub(r'\\[dt]?binom\{([^{}]*)\}\{([^{}]*)\}', r'C(\1,\2)', latex_src)
    latex_src = re.sub(r'\\boxed\{([^{}]*)\}', r'⟦\1⟧', latex_src)

    for macro, replacement in _PRE_LATEX_MACRO_FIXUPS.items():
        latex_src = latex_src.replace(macro, replacement)

    # ── Ensure spaces around math functions —───────────────────────
    # Prevents pylatexenc from fusing subscripts and functions (e.g. T_1\sin → T_1sin),
    # which would cause the bare subscript converter to mangle "1sin" into "₁ₛᵢₙ".
    latex_src = re.sub(
        r'\\(sin|cos|tan|cot|sec|csc|log|ln|lim|max|min|det|deg|sinh|cosh|tanh|arcsin|arccos|arctan)\b',
        r' \1 ', latex_src
    )

    # ── Extensible named arrows —————————————————────────——————
    # pylatexenc silently drops \xrightarrow / \xleftarrow (not in its table).
    # Convert here so the arrow survives. Use PARENS not brackets — Rich would
    # misparse [label] as a markup tag in the rendered output.
    latex_src = re.sub(r'\\xrightarrow\{([^}]*)\}', ' →(\\1) ', latex_src)
    latex_src = re.sub(r'\\xleftarrow\{([^}]*)\}',  ' ←(\\1) ', latex_src)
    latex_src = re.sub(r'\\xLeftrightarrow\{([^}]*)\}', ' ⇔(\\1) ', latex_src)
    latex_src = re.sub(r'\\xmapsto\{([^}]*)\}', ' ↦(\\1) ', latex_src)
    # \stackrel / \overset / \underset — pylatexenc drops these too
    latex_src = re.sub(r'\\(?:stackrel|overset)\{([^}]*)\}\{([^}]*)\}',
                       r'\2^{\1}', latex_src)
    latex_src = re.sub(r'\\underset\{([^}]*)\}\{([^}]*)\}',
                       r'\2_{\1}', latex_src)

    # ── Compound fraction exponents: ^{3/2} —————————————————————
    # Must run BEFORE _wrap_frac_args_in_parens so the exponent is already
    # clean Unicode when pylatexenc receives the wrapped expression.
    latex_src = re.sub(r'\^\{(\d+)/(\d+)\}', _fmt_frac_exp, latex_src)

    # ── Normalise display/text/continued fraction variants ───────────────────
    # pylatexenc silently swallows \dfrac, \tfrac, \cfrac (not in its table).
    # Normalise to \frac first so pylatexenc can process them correctly.
    latex_src = latex_src.replace(r'\dfrac', r'\frac')
    latex_src = latex_src.replace(r'\tfrac', r'\frac')
    latex_src = latex_src.replace(r'\cfrac', r'\frac')

    return latex_src



def _wrap_frac_args_in_parens(latex_src: str) -> str:
    """
    Rewrite every \\frac{NUM}{DEN} to (NUM)/(DEN), using a balanced-brace scan
    (not a single-level regex) so nested braces inside NUM/DEN — e.g. a
    fractional exponent like \\Delta^{3/2} — are extracted correctly instead
    of matching on the wrong inner '}'.

    Combined with _fmt_frac_exp (which runs first in _apply_pre_latex_fixups),
    \\frac{\\Delta^{3/2}}{6 A^2} now renders as "(Δ³⁄2)/(6A²)" — completely
    unambiguous. The old path without _fmt_frac_exp produced "(Δ³/2)/(6A²)" which
    still looked like Δ-cubed-over-2. _fmt_frac_exp eliminates that ambiguity
    by converting ^{3/2} to ³⁄2 (superscript-3 + fraction-slash + subscript-2)
    before this function or pylatexenc ever sees the expression.
    """
    def _extract_brace_group(s: str, start: int):
        """Return (content, end_index) for a {...} group starting at s[start],
        honoring nested braces. Returns None if malformed/not a brace group."""
        j = start
        while j < len(s) and s[j] in ' \t':
            j += 1
        if j >= len(s) or s[j] != '{':
            return None
        depth = 1
        k = j + 1
        while k < len(s) and depth > 0:
            if s[k] == '{':
                depth += 1
            elif s[k] == '}':
                depth -= 1
            k += 1
        if depth != 0:
            return None  # unbalanced — leave untouched rather than corrupt it
        return s[j + 1:k - 1], k

    out = []
    i, n = 0, len(latex_src)
    while i < n:
        idx = latex_src.find(r'\frac', i)
        if idx == -1:
            out.append(latex_src[i:])
            break
        out.append(latex_src[i:idx])
        pos = idx + len(r'\frac')

        num = _extract_brace_group(latex_src, pos)
        if num is None:
            out.append(r'\frac')
            i = pos
            continue
        num_content, pos_after_num = num
        den = _extract_brace_group(latex_src, pos_after_num)
        if den is None:
            out.append(r'\frac')
            i = pos
            continue
        den_content, pos_after_den = den

        # Recurse so nested \frac inside num/den is also expanded.
        num_content = _wrap_frac_args_in_parens(num_content)
        den_content = _wrap_frac_args_in_parens(den_content)
        out.append(f'({num_content})/({den_content})')
        i = pos_after_den
    return ''.join(out)

# ── Consistent style constants ───────────────────────────────────────────────
RULE_STYLE = "dim"
ACCENT     = "bold white"
META       = "dim"
WARN       = "yellow"
ERR        = "bold red"
SECTION    = "dim cyan"


# ══════════════════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def rule(label: str = "", style: str = RULE_STYLE):
    """Horizontal rule, expands to terminal width."""
    if label:
        console.rule(label, style=style)
    else:
        console.rule(style=style)


def _flatten_content(content) -> str:
    """Extract plain text from a chat message content field.

    Messages may store content as a plain string or as a list of
    dicts (vision turns).  This helper normalises both to a string.
    """
    if isinstance(content, list):
        return " ".join(b.get("text", "") for b in content if isinstance(b, dict))
    return content or ""


def show_image_in_terminal(image_path: str):
    """Render image inline using chafa if installed."""
    import shutil as _shutil
    import subprocess
    if _shutil.which("chafa"):
        try:
            # GIFs loop indefinitely in chafa by default, which makes
            # subprocess.run() block forever (turn never completes, history
            # never saved, next prompt never shown).
            # the timeout=15 is a
            # safety net — if
            # the process hasn't exited on its own, we kill it after 15 s.
            is_gif = image_path.lower().endswith(".gif")
            # Quality flags: 24-bit truecolor, all symbol types (incl. braille
            # for finest detail), Floyd-Steinberg dithering for smooth gradients.
            quality_flags = [
                "--colors", "full",    # 24-bit truecolor
                "--symbols", "all",    # braille + block + border = maximum detail
                "--dither", "fs",      # Floyd-Steinberg dithering
            ]
            cmd = ["chafa"] + quality_flags + [image_path]
            timeout = 15 if is_gif else None
            try:
                subprocess.run(cmd, timeout=timeout)
            except subprocess.TimeoutExpired:
                pass  # GIF played long enough; kill it and move on
            print()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE LOADING & ENHANCEMENT
# ══════════════════════════════════════════════════════════════════════════════

def _copy_and_enhance(item_path: str, filename: str) -> str:
    """
    Copy a local image file to scratch, show it in the terminal, and create
    an auto-enhanced copy for the AI to read.

    Returns the absolute path to the enhanced copy so the caller can either
    base64-encode it (vision path) or pass it to OCR (OCR path).

    The original file is never modified — only the enhanced copy is changed.
    """
    os.makedirs(config.SCRATCH_DIR, exist_ok=True)
    scratch_path = os.path.abspath(os.path.join(config.SCRATCH_DIR, filename))
    shutil.copy2(item_path, scratch_path)
    show_image_in_terminal(scratch_path)
    enhanced_path = os.path.join(config.SCRATCH_DIR, f"enhanced_{filename}")
    shutil.copy2(scratch_path, enhanced_path)
    auto_enhance(enhanced_path)
    return enhanced_path


def _build_master_enhanced(item_path: str, filename: str) -> str:
    """
    Produce a master enhanced image for the tiling pipeline.

    Applies (in order, via cv2):
      1. Saturation-guarded inversion  — only if dark AND greyscale
      2. Smart resize to 512–1536 px   — LANCZOS4
      3. LAB-space CLAHE               — local contrast, colour-safe
      4. Unsharp masking               — GaussianBlur addWeighted

    Falls back to a simpler PIL pipeline if cv2 is not installed.
    Returns the path to the master-enhanced file in scratch/.
    """
    os.makedirs(config.SCRATCH_DIR, exist_ok=True)
    master_path = os.path.join(config.SCRATCH_DIR, f"master_{filename}")
    shutil.copy2(item_path, master_path)

    # ── Fast path: IMAGE_ENHANCE=False → skip cv2 pipeline, PIL only ─────────
    if not getattr(config, "IMAGE_ENHANCE", True):
        auto_enhance(master_path)   # PIL: saturation-guarded inversion + contrast
        console.print(f"   [dim]master_enhance: PIL-only mode (IMAGE_ENHANCE=False)[/dim]")
        return master_path

    # ── Step 0a: EXIF orientation fix (via Pillow) ────────────────────────────
    # Phone cameras embed rotation in EXIF; cv2.imread ignores it, so a
    # sideways photo would be processed rotated. Pillow reads the tag and
    # applies the physical rotation before we hand the file to cv2.
    try:
        from PIL import Image as _PILImg
        from PIL import ExifTags as _ExifTags
        _pil = _PILImg.open(master_path)
        _exif = _pil.getexif() if hasattr(_pil, "getexif") else {}
        _orient_tag = next(
            (k for k, v in _ExifTags.TAGS.items() if v == "Orientation"), None
        )
        _orient = _exif.get(_orient_tag) if _orient_tag and _exif else None
        _ORIENT_MAP = {
            3: _PILImg.ROTATE_180,
            6: _PILImg.ROTATE_270,
            8: _PILImg.ROTATE_90,
        }
        if _orient in _ORIENT_MAP:
            # Bug 29: call .load() to force Pillow to fully read pixel data into
            # memory and release the OS file handle before we overwrite the same
            # path.  Without this, Image.open() holds the file open lazily and
            # save() fails with PermissionError on Windows.
            _pil.load()
            _pil = _pil.transpose(_ORIENT_MAP[_orient])
            _pil.save(master_path)
            console.print(f"   [dim cyan]master_enhance: EXIF orientation corrected (tag={_orient})[/dim cyan]")
    except Exception:
        pass   # silently skip if PIL unavailable or EXIF unreadable

    try:
        import cv2
        import numpy as np

        img = cv2.imread(master_path, cv2.IMREAD_COLOR)
        if img is None:
            return master_path

        # ── Step 0b: Smart margin trimming ────────────────────────────────────
        # Detects the bounding box of actual page content (ink, lines) and
        # crops away surrounding clutter (thumb, carpet, table surface).
        # Safety margin: we keep 92 % of the detected box to avoid clipping
        # edge content. Skipped if the box is < 10 % smaller than original
        # (nothing worth trimming) or if contour detection gives garbage.
        try:
            _h0, _w0 = img.shape[:2]
            _gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _blur    = cv2.GaussianBlur(_gray, (5, 5), 0)
            _, _thresh = cv2.threshold(_blur, 0, 255,
                                       cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            _cnts, _ = cv2.findContours(_thresh, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
            if _cnts:
                _bx, _by, _bw, _bh = cv2.boundingRect(
                    max(_cnts, key=cv2.contourArea)
                )
                # Safety margin: expand detected box by 4 % each side capped at image edge
                _pad_x = int(_bw * 0.04)
                _pad_y = int(_bh * 0.04)
                _x1 = max(0, _bx - _pad_x)
                _y1 = max(0, _by - _pad_y)
                _x2 = min(_w0, _bx + _bw + _pad_x)
                _y2 = min(_h0, _by + _bh + _pad_y)
                _crop_w, _crop_h = _x2 - _x1, _y2 - _y1
                # Skip if crop is <10 % smaller (no meaningful margin to remove)
                if _crop_w < _w0 * 0.90 or _crop_h < _h0 * 0.90:
                    img = img[_y1:_y2, _x1:_x2]
                    console.print(
                        f"   [dim cyan]master_enhance: margin trimmed "
                        f"{_w0}×{_h0} → {_crop_w}×{_crop_h}[/dim cyan]"
                    )
        except Exception:
            pass   # silently skip on any contour failure

        # ── 1. Saturation-guarded inversion ───────────────────────────────────
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        avg_v = float(hsv[:, :, 2].mean())   # brightness  0-255
        avg_s = float(hsv[:, :, 1].mean())   # saturation  0-255
        if avg_v < 80 and avg_s < 30:        # dark + greyscale → invert
            img = 255 - img
            console.print(f"   [dim cyan]master_enhance: inverted (dark+greyscale)[/dim cyan]")

        # ── 2. Smart resize to 512–1536 px range ──────────────────────────────
        h, w = img.shape[:2]
        scale = 1.0
        if max(h, w) > 1536:
            scale = 1536 / max(h, w)
        elif min(h, w) < 512:
            scale = 512 / min(h, w)
        if scale != 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            console.print(f"   [dim cyan]master_enhance: resized {w}×{h} → {new_w}×{new_h}[/dim cyan]")

        # ── 3. LAB-space CLAHE with bleed-through suppression ─────────────────
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)

        # Document-like images (low saturation — handwritten notes, photocopies,
        # dark-mode screenshots) get percentile clipping BEFORE CLAHE to erase
        # back-page bleed-through. The 5th–80th percentile stretch forces the
        # faint mid-gray bleed (intensity ~140-170) and lighter paper to pure white
        # so CLAHE never mistakes it for low-contrast real ink and boosts it.
        # Natural photos (avg_s >= 60) skip this — it would destroy mid-tones.
        if avg_s < 60:
            p_dark, p_light = np.percentile(l_ch, (5, 80))
            if p_light > p_dark:   # guard against degenerate (all-black) images
                l_ch = np.clip(
                    (l_ch.astype(np.float32) - p_dark) * (255.0 / (p_light - p_dark + 1e-5)),
                    0, 255
                ).astype(np.uint8)
                console.print(
                    f"   [dim cyan]master_enhance: bleed suppression "
                    f"(p5={p_dark:.0f} → black, p80={p_light:.0f} → white)[/dim cyan]"
                )

            # Gentler CLAHE — p80 percentile stretch did the heavy lifting;
            # lower clipLimit prevents accidentally amplifying any ghost text.
            clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))

        else:
            # Natural photo — full CLAHE, no percentile clipping
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        l_ch = clahe.apply(l_ch)
        img = cv2.cvtColor(cv2.merge((l_ch, a_ch, b_ch)), cv2.COLOR_LAB2BGR)

        # ── 4. Unsharp mask (stronger for document crispness) ─────────────────
        # Weights (1.8, -0.8): output = img + 0.8 × high-freq detail.
        # Gives ink a sharp, dark bite — equivalent to bold ink — without
        # the stroke-merging artefacts that morphological erosion causes.
        gaussian = cv2.GaussianBlur(img, (0, 0), 3.0)
        img = cv2.addWeighted(img, 1.8, gaussian, -0.8, 0)


        _ext = os.path.splitext(master_path)[1].lower()
        if _ext in (".jpg", ".jpeg"):
            _write_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
        elif _ext == ".png":
            _write_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
        else:
            _write_params = []
        cv2.imwrite(master_path, img, _write_params)
        h2, w2 = img.shape[:2]
        console.print(f"   [dim cyan]master_enhance: cv2 pipeline done ({w2}×{h2})[/dim cyan]")

    except ImportError:
        # ── PIL fallback ──────────────────────────────────────────────────────
        try:
            from PIL import Image, ImageOps, ImageEnhance
            import numpy as np

            img_pil = Image.open(master_path).convert("RGB")
            arr = np.array(img_pil, dtype=np.float32)

            # Simplified saturation check via numpy
            arr_norm = arr / 255.0
            max_c = arr_norm.max(axis=2)
            min_c = arr_norm.min(axis=2)
            sat_map = np.where(max_c > 0, (max_c - min_c) / max_c, 0.0)

            if arr.mean() < 80 and sat_map.mean() < 0.12:
                img_pil = ImageOps.invert(img_pil)

            img_pil = ImageEnhance.Contrast(img_pil).enhance(1.3)
            img_pil.save(master_path, quality=95)
            console.print(f"   [dim cyan]master_enhance: PIL fallback applied[/dim cyan]")
        except Exception as pil_err:
            console.print(f"   [dim yellow]master_enhance fallback failed ({pil_err})[/dim yellow]")

    return master_path


def _slice_master_into_tiles(master_path: str, stem: str) -> list:
    """
    Slice a master enhanced image into 5 images:
      - overview_<stem>.jpg      : full image, downscaled ≤1024px, letterboxed to square
      - tile_top_left_<stem>.jpg : top-left crop   (55 % × 55 % of master, 10 % overlap)
      - tile_top_right_<stem>.jpg
      - tile_bottom_left_<stem>.jpg
      - tile_bottom_right_<stem>.jpg

    Returns a list of {filename, src, mime} dicts (same shape as load_images_for_vision).
    Returns [] if the master is too small to tile meaningfully (< 500 px on shorter side)
    or if neither cv2 nor PIL is available.
    """
    # ── Try cv2 first, then fall back to PIL ──────────────────────────────────
    try:
        import cv2
        import numpy as np
        _use_cv2 = True
    except ImportError:
        _use_cv2 = False
        try:
            from PIL import Image as _PILImage
        except ImportError:
            return []

    results = []

    if _use_cv2:
        master = cv2.imread(master_path, cv2.IMREAD_COLOR)
        if master is None:
            return []
        h, w = master.shape[:2]
    else:
        _pil_master = _PILImage.open(master_path).convert("RGB")
        w, h = _pil_master.size

    # Tiling threshold: tiles must be meaningful in size
    if min(h, w) < 500:
        console.print(f"   [dim]tiling skipped: master too small ({w}×{h} px)[/dim]")
        return []

    def _encode_file(path: str) -> str:
        with open(path, "rb") as _f:
            return base64.b64encode(_f.read()).decode("utf-8")

    # ── Overview: downscale to ≤1024px then letterbox to square ──────────────
    ov_fname = f"overview_{stem}.jpg"
    ov_path  = os.path.join(config.SCRATCH_DIR, ov_fname)

    if _use_cv2:
        ov_scale = min(1024 / max(h, w), 1.0)
        ov_w, ov_h = int(w * ov_scale), int(h * ov_scale)
        overview = cv2.resize(master, (ov_w, ov_h), interpolation=cv2.INTER_LANCZOS4)
        side = max(ov_w, ov_h)
        canvas = np.full((side, side, 3), 255, dtype=np.uint8)
        canvas[(side - ov_h) // 2:(side - ov_h) // 2 + ov_h,
               (side - ov_w) // 2:(side - ov_w) // 2 + ov_w] = overview
        _ov_ext = os.path.splitext(ov_path)[1].lower()
        _ov_params = [cv2.IMWRITE_JPEG_QUALITY, 90] if _ov_ext in (".jpg", ".jpeg") else ([cv2.IMWRITE_PNG_COMPRESSION, 3] if _ov_ext == ".png" else [])
        cv2.imwrite(ov_path, canvas, _ov_params)
    else:
        ov_scale = min(1024 / max(h, w), 1.0)
        ov_w, ov_h = int(w * ov_scale), int(h * ov_scale)
        overview = _pil_master.resize((ov_w, ov_h), _PILImage.LANCZOS)
        side = max(ov_w, ov_h)
        canvas_pil = _PILImage.new("RGB", (side, side), (255, 255, 255))
        canvas_pil.paste(overview, ((side - ov_w) // 2, (side - ov_h) // 2))
        canvas_pil.save(ov_path, quality=90)

    results.append({"filename": ov_fname, "src": f"data:image/jpeg;base64,{_encode_file(ov_path)}", "mime": "image/jpeg"})
    console.print(f"   [dim cyan]tile overview: {ov_w}×{ov_h} → {max(ov_w,ov_h)}×{max(ov_w,ov_h)} (letterboxed)[/dim cyan]")

    # ── Tiles: configurable grid with 10 % overlap ────────────────────────────
    # Grid shape comes from config.TILING_GRID = (rows, cols).
    # Each tile spans 1/N of the axis + 5 % padding each side (10 % overlap).
    rows, cols = getattr(config, "TILING_GRID", (2, 2))

    _ROW_NAMES = {1: ["center"], 2: ["top", "bottom"], 3: ["top", "middle", "bottom"]}
    _COL_NAMES = {1: ["center"], 2: ["left", "right"],  3: ["left", "center", "right"]}
    row_names = _ROW_NAMES.get(rows, [f"row{i}" for i in range(rows)])
    col_names = _COL_NAMES.get(cols, [f"col{i}" for i in range(cols)])

    tile_defs = []
    for r in range(rows):
        for c in range(cols):
            x1 = max(0, int(c * w / cols       - w * 0.05))
            y1 = max(0, int(r * h / rows       - h * 0.05))
            x2 = min(w, int((c + 1) * w / cols + w * 0.05))
            y2 = min(h, int((r + 1) * h / rows + h * 0.05))
            tile_defs.append((f"{row_names[r]}_{col_names[c]}", x1, y1, x2, y2))

    for name, x1, y1, x2, y2 in tile_defs:
        t_fname = f"tile_{name}_{stem}.jpg"
        t_path  = os.path.join(config.SCRATCH_DIR, t_fname)

        if _use_cv2:
            tile = master[y1:y2, x1:x2]
            _t_ext = os.path.splitext(t_path)[1].lower()
            _t_params = [cv2.IMWRITE_JPEG_QUALITY, 95] if _t_ext in (".jpg", ".jpeg") else ([cv2.IMWRITE_PNG_COMPRESSION, 3] if _t_ext == ".png" else [])
            cv2.imwrite(t_path, tile, _t_params)
        else:
            tile_pil = _pil_master.crop((x1, y1, x2, y2))
            tile_pil.save(t_path, quality=95)

        results.append({"filename": t_fname, "src": f"data:image/jpeg;base64,{_encode_file(t_path)}", "mime": "image/jpeg"})
        console.print(f"   [dim cyan]tile {name}: {x2 - x1}×{y2 - y1}[/dim cyan]")

    return results



def load_images_for_vision(image_input: str) -> list:
    """
    Parse image_input, encode local files as base64 data URIs, return list of:
        {"filename": str, "src": str, "mime": str}

    src is a data URI for local files or a plain https URL for remote images.
    Used when config.SUPPORTS_VISION is True — images go directly to the model.
    """
    if not image_input.strip():
        return []

    try:
        items = shlex.split(image_input, posix=False)
    except Exception:
        items = image_input.split()

    items = [i.strip().strip('"').strip("'") for i in items if i.strip()]
    results = []

    for item in items:
        filename = Path(item).name if not item.startswith("http") else item.split("/")[-1]
        try:
            if item.startswith("http://") or item.startswith("https://"):
                results.append({"filename": filename, "src": item, "mime": "image/jpeg"})
            elif item.startswith("data:"):
                results.append({"filename": filename, "src": item, "mime": "image/jpeg"})
            elif os.path.isfile(item):
                ext  = Path(item).suffix.lower()
                mime = MIME_MAP.get(ext, "image/jpeg")
                size_mb = os.path.getsize(item) / (1024 * 1024)
                console.print(f"   [dim]loading {filename} ({size_mb:.2f} MB)[/dim]")

                # ── Tiling path (IMAGE_TILING = True) ─────────────────────────────
                if config.IMAGE_TILING:
                    show_image_in_terminal(item)
                    stem = Path(item).stem
                    try:
                        master_path = _build_master_enhanced(item, filename)
                        tile_dicts  = _slice_master_into_tiles(master_path, stem)
                        if tile_dicts:
                            results.extend(tile_dicts)
                            continue   # skip standard path for this image
                        else:
                            console.print(f"   [dim yellow]tiling produced no tiles — using standard path[/dim yellow]")
                    except Exception as tile_err:
                        console.print(f"   [dim yellow]tiling failed ({tile_err}) — using standard path[/dim yellow]")

                # ── Standard path (tiling off, or tiling fell through) ─────────
                with open(item, "rb") as f:
                    raw = f.read()
                encoded = base64.b64encode(raw).decode("utf-8")

                try:
                    enhanced_path = _copy_and_enhance(item, filename)
                    with open(enhanced_path, "rb") as _f:
                        raw = _f.read()
                    encoded = base64.b64encode(raw).decode("utf-8")
                except Exception as copy_err:
                    console.print(f"   [dim yellow]image enhancement failed ({copy_err}) — using original[/dim yellow]")

                results.append({
                    "filename":     filename,
                    "src":          f"data:{mime};base64,{encoded}",
                    "mime":         mime,
                })
            else:
                console.print(f"   [yellow]image not found: {item}[/yellow]")
        except Exception as e:
            console.print(f"   [yellow]could not load {item}: {e}[/yellow]")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# OCR IMAGE PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def process_images_via_ocr(image_input: str, ocr_agent) -> str:
    """
    Run each image through OCR, print results, return formatted text blocks
    ready to embed in the prompt.

    Mirrors load_images_for_vision()'s preprocessing pipeline:
      - IMAGE_TILING=True  → _build_master_enhanced + _slice_master_into_tiles
        → ocr_agent.process_image_group() (one API call with all tiles per image)
      - IMAGE_TILING=False → _copy_and_enhance (PIL auto-enhance)
        → ocr_agent.process_images() (one API call per image, independently)

    ocr_agent: an ImageOCRAgent instance passed by the caller (start.py).
    The original file is never modified — only enhanced copies are used.
    """
    # Bug #14 fix: track all temp image files created by enhancement/tiling
    # and delete them when the OCR pass finishes, preventing scratch directory leak.
    _temp_files = []
    
    if not image_input:
        return ""

    try:
        _items = [i.strip().strip('"').strip("'")
                  for i in shlex.split(image_input, posix=False) if i.strip()]
    except Exception:
        _items = image_input.split()

    if not _items:
        return ""

    ocr_results = []

    if getattr(config, "IMAGE_TILING", False):
        # ── Tiling path ────────────────────────────────────────────────────────
        # Each source image is enhanced → tiled → sent to OCR as ONE group call
        # (overview + crops in one API message) → one result dict per image.
        # Mirrors the vision path: model sees full context + zoomed detail at once.
        for _item in _items:
            if (_item.startswith("http://") or _item.startswith("https://")
                    or _item.startswith("data:")):
                # URLs / data-URIs: can't tile, send as-is
                try:
                    _res = ocr_agent.process_images(f'"{_item}"')
                    ocr_results.extend(_res)
                except Exception as _e:
                    console.print(f"   [yellow]⚠️  OCR failed for '{_item[:60]}': {_e}[/yellow]")
                continue

            if not os.path.isfile(_item):
                console.print(f"   [yellow]image not found: {_item}[/yellow]")
                continue

            _filename = Path(_item).name
            _stem     = Path(_item).stem
            _size_mb  = os.path.getsize(_item) / (1024 * 1024)
            console.print(f"   [dim]loading {_filename} ({_size_mb:.2f} MB)[/dim]")
            show_image_in_terminal(_item)

            try:
                _master = _build_master_enhanced(_item, _filename)
                _temp_files.append(_master)
                _tiles  = _slice_master_into_tiles(_master, _stem)
                if _tiles:
                    _tile_paths = []
                    for d in _tiles:
                        _tp = os.path.join(config.SCRATCH_DIR, d["filename"])
                        _temp_files.append(_tp)
                        if os.path.isfile(_tp):
                            _tile_paths.append(_tp)
                    if _tile_paths:
                        _ocr_input = " ".join(f'"{p}"' for p in _tile_paths)
                        _res = ocr_agent.process_image_group(_ocr_input)
                        _res["source"] = _filename   # label with original filename
                        ocr_results.append(_res)
                        continue
                # Tiling produced no tiles — fall back to master image
                console.print(f"   [dim yellow]tiling produced no tiles — OCR'ing master image[/dim yellow]")
                _res = ocr_agent.process_images(f'"{_master}"')
                ocr_results.extend(_res)
            except Exception as _tile_err:
                console.print(f"   [dim yellow]tiling failed ({_tile_err}) — using standard enhance[/dim yellow]")
                try:
                    _enhanced = _copy_and_enhance(_item, _filename)
                    _temp_files.append(_enhanced)
                    _res = ocr_agent.process_images(f'"{_enhanced}"')
                    ocr_results.extend(_res)
                except Exception as _e:
                    console.print(f"   [yellow]⚠️  OCR failed for '{_filename}': {_e}[/yellow]")

    else:
        # ── Standard path (IMAGE_TILING=False) ────────────────────────────────
        # _copy_and_enhance for each file, then OCR all independently.
        # Unchanged from original behaviour.
        try:
            _enhanced_inputs = []
            for _item in _items:
                if (not _item.startswith("http") and
                        not _item.startswith("data:") and
                        os.path.isfile(_item)):
                    try:
                        _enhanced_path = _copy_and_enhance(_item, Path(_item).name)
                        _temp_files.append(_enhanced_path)
                        _enhanced_inputs.append(f'"{_enhanced_path}"')
                    except Exception:
                        _enhanced_inputs.append(f'"{_item}"')
                else:
                    _enhanced_inputs.append(f'"{_item}"')
            _combined = " ".join(_enhanced_inputs)
        except Exception:
            _combined = image_input   # non-fatal fallback

        ocr_results = ocr_agent.process_images(_combined)

    if not ocr_results:
        return ""

    rule("ocr results", style=SECTION)

    prompt_blocks = []
    for r in ocr_results:
        label    = r["label"]
        source   = r["source"]
        markdown = r["markdown"]

        console.print(f"\n[{ACCENT}]{label}[/{ACCENT}] [{META}]{source[:70]}[/{META}]")
        console.print(Markdown(markdown))

        prompt_blocks.append(
            f"---\n"
            f"[ATTACHED IMAGE — {label} — transcribed via vision pipeline (may have minor inconsistencies)]\n"
            f"{markdown}\n"
            f"---"
        )

    for p in _temp_files:
        try:
            if os.path.isfile(p):
                os.remove(p)
        except Exception:
            pass

    rule(style=SECTION)
    return "\n\n".join(prompt_blocks)



# ══════════════════════════════════════════════════════════════════════════════
# LaTeX → UNICODE CONVERTER
# ══════════════════════════════════════════════════════════════════════════════

def prep_for_console(text: str) -> str:
    """Convert LaTeX and special notation to Unicode for terminal display."""
    # ── Fast path: skip all processing for plain prose / code ──────────
    # The overwhelming majority of responses contain no math at all.
    # A single linear scan to check for trigger characters is far cheaper
    # than running 30+ regexes that will all find zero matches.
    if not isinstance(text, str) or not text:
        return text or ""
    _math_triggers = ('\\', '$', '^{', '_{', '^', '\\frac', '\\sum', '\\int',
                      '\\sqrt', '\\alpha', '\\beta', '\\theta', '\\pi',
                      '\\rightarrow', '\\leftarrow', '\\Rightarrow',
                      '\\[', '\\(', '~')
    # Bug 12: also catch bare numeric subscripts like v_1 or T_2 that have no
    # LaTeX markers or $-delimiters, so the fast-path doesn't skip them.
    _has_bare_subscript = bool(re.search(r'[A-Za-z]_\d', text))
    if not any(t in text for t in _math_triggers) and not _has_bare_subscript:
        return text

    # ── 0. Stash inline code spans so regexes below never touch them ──
    # Single-backtick spans: `...`  (triple-backtick blocks are already
    # handled by print_smart_response before we're called, so we only
    # need to guard single-backtick inline code here.)
    _stash: list[str] = []
    _PLACEHOLDER = "\x00CODESTASH{}\x00"

    def _stash_code(m: re.Match) -> str:
        _stash.append(m.group(0))
        return _PLACEHOLDER.format(len(_stash) - 1)

    text = re.sub(r'``[^`\n]+``', _stash_code, text)  # double-backtick spans FIRST
    text = re.sub(r'`[^`\n]+`', _stash_code, text)
    # Stash XML tags so things like <bash>echo $PATH</bash> don't become math blocks
    text = re.sub(r'<(\w+)(?:\s[^>]*)?>.*?</\1>', _stash_code, text, flags=re.DOTALL)

    # ── 1. $$ and $ blocks via pylatexenc (DO THIS FIRST!) ─────────
    def replace_math(m):
        try:
            src = m.group(1)
            src = _apply_pre_latex_fixups(src)
            src = _wrap_frac_args_in_parens(src)
            result = _latex_converter.latex_to_text(src)
            return result.replace('|', '∣')  # prevent | from breaking table columns
        except Exception:
            return m.group(0) # If it fails, return the original string untouched

    # \[...\] and \(...\) — model sometimes outputs these instead of $$ / $
    text = re.sub(r'\\\[(.+?)\\\]', lambda m: f'\n{replace_math(m)}\n', text, flags=re.DOTALL)
    text = re.sub(r'\\\((.+?)\\\)', replace_math, text, flags=re.DOTALL)

    text = re.sub(r'\$\$(.+?)\$\$', lambda m: f'\n{replace_math(m)}\n', text, flags=re.DOTALL)

    # Safe inline math: won't trigger on currency like "$5 and $10".
    # [^\n$]+? keeps the match on a single line and won't span across $ signs,
    # preventing accidental multi-expression merges.
    text = re.sub(r'(?<!\\)\$(?!\s)([^\n$]+?)(?<!\s)\$', replace_math, text)

    # ── 2. Named arrows (outside math blocks) ──────────────────────
    # Use PARENS not brackets — Rich misparses [label] as a markup tag.
    text = re.sub(r'\\xrightarrow\{([^}]+)\}', ' →(\\1) ', text)
    text = re.sub(r'\\xleftarrow\{([^}]+)\}',  ' ←(\\1) ', text)
    text = re.sub(r'\\xLeftrightarrow\{([^}]+)\}', ' ⇔(\\1) ', text)
    text = re.sub(r'\\xmapsto\{([^}]+)\}', ' ↦(\\1) ', text)
    text = re.sub(r'\\(?:stackrel|overset)\{([^}]*)\}\{([^}]*)\}', '\\2(\\1)', text)
    text = re.sub(r'\\underset\{([^}]*)\}\{([^}]*)\}', '\\2(\\1)', text)

    # ── 2.1. \not-prefixed negations ──────────────────────────────
    # Must run BEFORE the bare dict so \not\in isn't split into negation + membership.
    # ORDERING: longer/more-specific forms FIRST (\not\subseteq before \not\subset).
    _NOT_MAP = (
        (r'\not\in',       '\u2209'),  (r'\not\notin',    '\u2208'),  # double-neg
        (r'\not\subseteq', '\u2288'),  (r'\not\supseteq', '\u2289'),
        (r'\not\subset',   '\u2284'),  (r'\not\supset',   '\u2285'),
        (r'\not\equiv',    '\u2262'),  (r'\not\approx',   '\u2249'),
        (r'\not\cong',     '\u2247'),  # \not\cong BEFORE shorter \not\ forms
        (r'\not\simeq',    '\u2244'),  (r'\not\sim',      '\u2241'),
        # \not\parallel BEFORE bare dict converts \parallel
        (r'\not\parallel', '\u2226'),  (r'\not\perp',     '\u2aeb'),
        # \not\subset etc already above; negated orders also above ─
        (r'\not\leq',      '\u2270'),  (r'\not\geq',      '\u2271'),
        (r'\not\le',       '\u2270'),  (r'\not\ge',       '\u2271'),
        (r'\not\prec',     '\u2280'),  (r'\not\succ',     '\u2281'),
        (r'\not=',         '\u2260'),  (r'\not<',         '\u226e'),
        (r'\not>',         '\u226f'),
    )
    for _nm, _nr in _NOT_MAP:
        text = text.replace(_nm, _nr)

    # ── 2.2. \left / \right delimiter cleanup ──────────────
    # Models frequently write these outside math delimiters. Map to plain
    # ASCII brackets — mathematically identical, no LaTeX garbage.
    _DELIM_MAP = (
        # ─ Curly braces ──────────────────────────────────────────────────
        (r'\left\{',    '{'),   (r'\right\}',   '}'),   # braces FIRST
        # ─ Angle brackets — MUST precede \left( so \left\langle is handled whole ─
        (r'\left\langle','\u27e8'), (r'\right\rangle','\u27e9'),
        (r'\left\|',    '\u2016'),  (r'\right\|',    '\u2016'),  # double-bar norm
        # ─ Ordinary brackets ──────────────────────────────────────────────
        (r'\left(',     '('),   (r'\right)',     ')'),
        (r'\left[',     '['),   (r'\right]',     ']'),
        (r'\left|',     '|'),   (r'\right|',     '|'),
        (r'\left.',     ''),    (r'\right.',     ''),
        (r'\langle',   '\u27e8'), (r'\rangle',   '\u27e9'),  # standalone angle
        (r'\bigl(',     '('),   (r'\bigr)',      ')'),
        (r'\bigl[',     '['),   (r'\bigr]',      ']'),
        (r'\Bigl(',     '('),   (r'\Bigr)',      ')'),
        (r'\Bigl[',     '['),   (r'\Bigr]',      ']'),
        (r'\Big(',      '('),   (r'\Big)',       ')'),
        (r'\big(',      '('),   (r'\big)',       ')'),
        (r'\big|',      '|'),   (r'\Big|',       '|'),
        (r'\vert',      '|'),   (r'\Vert',       '\u2016'),
        # ─ Escaped braces (set builder / literal) ────────────────────────
        (r'\{',         '{'),   (r'\}',          '}'),
        # ─ Norm \|..\| ────────────────────────────────────────
        (r'\|',         '\u2016'),  # double-bar norm; after \left\| so compound matches first
    )
    for _dm, _dr in _DELIM_MAP:
        text = text.replace(_dm, _dr)

    # ── 2.3. Math decorator stripping ──────────────
    # \mathbf{F}, \mathit{x}, \mathrm{d}, \mathcal{L}, \text{if x>0}
    # Strip the wrapper, keep the content. Loses font style (irrelevant
    # in terminal) but preserves the mathematical symbol exactly.
    # NOTE: \mathbb is intentionally EXCLUDED here — specific double-struck
    # letters (\mathbb{R}→ℝ etc.) are handled by _bare in section 5.
    text = re.sub(r'\\math(?:bf|it|rm|cal|sf|tt|scr|frak|op|rel|bin|ord)\{([^{}]*)\}', r'\1', text)
    text = re.sub(r'\\(?:text|mbox|hbox)\{([^{}]*)\}', r'\1', text)
    text = re.sub(r'\\operatorname\*?\{([^{}]*)\}', r'\1', text)
    text = re.sub(r'\\(?:textbf|textit|emph)\{([^{}]*)\}', r'\1', text)
    text = re.sub(r'\\boldsymbol\{([^{}]*)\}', r'\1', text)   # \boldsymbol{\alpha} → \alpha
    text = re.sub(r'\\ensuremath\{([^{}]*)\}', r'\1', text)   # \ensuremath{x^2} → x^2
    # Style switches that never carry content — simply erase them
    text = re.sub(r'\\(?:displaystyle|textstyle|scriptstyle|scriptscriptstyle)\b', '', text)

    # ── 2.4. Binomial coefficients ─────────────────────────────────
    text = re.sub(r'\\[dt]?binom\{([^{}]*)\}\{([^{}]*)\}', r'C(\1,\2)', text)

    # ── 2.5. Bracket decorators ────────────────────────────────────
    text = re.sub(r'\\underbrace\{([^{}]+)\}_\{([^{}]+)\}', r'\1 [\2]', text)
    text = re.sub(r'\\overbrace\{([^{}]+)\}\^\{([^{}]+)\}',  r'\1 [\2]', text)
    text = re.sub(r'\\underbrace\{([^{}]+)\}', r'\1', text)
    text = re.sub(r'\\overbrace\{([^{}]+)\}',  r'\1', text)
    text = re.sub(r'\\substack\{([^{}]+)\}',
                  lambda m: m.group(1).replace('\\\\', ', '), text)
    text = re.sub(r'\\cancel\{([^{}]*)\}', r'\1', text)
    text = re.sub(r'\\boxed\{([^{}]*)\}', '⟦\\1⟧', text)
    text = re.sub(r'\\color\{[^{}]*\}\{([^{}]*)\}', r'\1', text)
    text = re.sub(r'\\textcolor\{[^{}]*\}\{([^{}]*)\}', r'\1', text)

    # ── 2.6. LaTeX environments ────────────────────────────────────
    def _clean_env(m: re.Match) -> str:
        env  = m.group(1)
        body = m.group(2)
        # ─ Strip column spec from tabular/array-like envs ─────────────────────
        body = re.sub(r'^\s*\{[^{}]*\}', '', body)
        # ─ Strip structural annotation macros that have no visual meaning ──
        body = re.sub(r'\\cline\{[^{}]*\}', '', body)
        body = re.sub(r'\\label\{[^{}]*\}', '', body)
        body = re.sub(r'\\tag\{[^{}]*\}', '', body)
        body = re.sub(r'\\tag\*\{[^{}]*\}', '', body)
        body = body.replace('\\notag', '').replace('\\nonumber', '')
        # ─ \item → newline + bullet inside list environments ───────────────────
        if env in ('itemize', 'enumerate', 'description', 'compactitem', 'compactenum'):
            body = re.sub(r'\\item\s*\[([^\]]+)\]\s*', '\n\u2022 [\\1] ', body)  # \item[label]
            body = re.sub(r'\\item\s*', '\n\u2022 ', body)                        # \item
        else:
            body = re.sub(r'\\item\s*\[([^\]]+)\]\s*', '\n\u2022 [\\1] ', body)
            body = re.sub(r'\\item\s*', '\n\u2022 ', body)
        # ─ Format body into readable form ──────────────────────────────────
        body = (body
                .replace('\\\\', '\n')
                .replace('&', '  ')
                .replace('\\hline', '')
                .strip())
        if env in ('pmatrix', 'bmatrix', 'Bmatrix', 'vmatrix', 'Vmatrix', 'matrix'):
            return f'[{body}]'
        if env in ('cases', 'dcases'):
            return '{ ' + body
        return body
    # Run _clean_env repeatedly to handle nested environments (inner first).
    # Each pass reduces one level of nesting. Stop when text stabilises.
    for _ in range(3):
        new = re.sub(r'\\begin\{(\w+\*?)\}(.*?)\\end\{\1\}',
                     _clean_env, text, flags=re.DOTALL)
        if new == text:
            break
        text = new

    # ── 2.7. LaTeX spacing macros + structural noise ─────────────────
    for _sp, _sr in (
        (r'\qquad', '    '), (r'\quad', '  '),
        (r'\thinspace', ' '), (r'\medspace', ' '), (r'\thickspace', ' '),
        (r'\,', ' '), (r'\;', ' '), (r'\:', ' '), (r'\!', ''), (r'\ ', ' '),
        # Skip spacing macros — handled later with individual regex
        (r'\bigskip', '\n'), (r'\medskip', ' '), (r'\smallskip', ' '),
        # Table/structure commands that don't map to any visible character
        (r'\hline', ''), (r'\toprule', ''), (r'\midrule', ''), (r'\bottomrule', ''),
    ):
        text = text.replace(_sp, _sr)
    # \cline{m-n} in bare text (outside environments)
    text = re.sub(r'\\cline\{[^{}]*\}', '', text)
    # \item in bare text (outside list environments) → bullet point
    text = re.sub(r'\\item\s*\[([^\]]+)\]\s*', '\n\u2022 [\\1] ', text)   # \item[label]
    text = re.sub(r'\\item\s*', '\n\u2022 ', text)                          # bare \item
    # ── Strip annotation/measurement macros with no terminal meaning ──────────────
    # These produce zero visible output in LaTeX; strip to avoid raw \cmd leaking.
    text = re.sub(r'\\label\{[^{}]*\}', '', text)          # \label{eq:1}
    text = re.sub(r'\\tag\*?\{[^{}]*\}', '', text)         # \tag{*} \tag*{*}
    text = text.replace('\\notag', '').replace('\\nonumber', '')
    # ── Spacing / box macros: strip wrapper, PRESERVE content ───────────────────
    text = re.sub(r'\\phantom\{([^{}]*)\}', '', text)      # invisible — drop entirely
    text = re.sub(r'\\vphantom\{([^{}]*)\}', '', text)     # invisible vertical
    text = re.sub(r'\\hphantom\{([^{}]*)\}', '', text)     # invisible horizontal
    text = re.sub(r'\\smash\{([^{}]*)\}', r'\1', text)     # zero-height — keep content
    # \vspace / \hspace — strip entirely (terminal doesn't do explicit spacing)
    text = re.sub(r'\\vspace\*?\{[^{}]*\}', '', text)
    text = re.sub(r'\\hspace\*?\{[^{}]*\}', ' ', text)     # hspace → single space
    # \rule{width}{height} — strip (no way to draw a rule in terminal)
    text = re.sub(r'\\rule\{[^{}]*\}\{[^{}]*\}', '', text)
    # \kern / \mkern — strip spacing adjustments (including negative: \kern -1em)
    text = re.sub(r'\\m?kern\s*-?\s*[\d.]*\s*(?:pt|em|ex|mu|cm|mm|in)?', '', text)
    # \bm{x} bold math — strip wrapper, keep content (like \mathbf)
    text = re.sub(r'\\bm\{([^{}]*)\}', r'\1', text)
    # ── Structural / formatting no-ops: strip before _bare to prevent prefix eating ──
    # CRITICAL ORDER: \intertext before \int, \caption before \cap, etc.
    # These produce NO output in terminal — content preserved where applicable.
    text = re.sub(r'\\intertext\{([^{}]*)\}',      r' \1 ', text)  # \intertext
    text = re.sub(r'\\shortintertext\{([^{}]*)\}', r' \1 ', text)  # \shortintertext
    # ── Color wrappers: strip color metadata, preserve content ─────────────────────
    text = re.sub(r'\\colorbox\{[^{}]*\}\{([^{}]*)\}', r'\1', text)           # \colorbox
    text = re.sub(r'\\fcolorbox\{[^{}]*\}\{[^{}]*\}\{([^{}]*)\}', r'\1', text) # \fcolorbox
    # ── Content-bearing commands: strip wrapper keep content ────────────────────
    text = re.sub(r'\\footnote\{([^{}]*)\}', r' [\1]', text)   # footnote → inline [note]
    text = re.sub(r'\\footnotetext\{([^{}]*)\}', r' [\1]', text)
    text = re.sub(r'\\caption\*?\{([^{}]*)\}', r'\1', text)    # caption content visible
    text = re.sub(r'\\ensuremath\{([^{}]*)\}', r'\1', text)    # inline math wrapper
    # ── Layout macros: strip entirely (no visible effect in terminal) ───────────
    for _lm in (r'\centering', r'\raggedright', r'\raggedleft',
                r'\noindent', r'\allowdisplaybreaks',
                r'\displaybreak', r'\strut', r'\mathstrut',
                r'\newline', r'\linebreak', r'\pagebreak', r'\clearpage', r'\newpage'):
        text = text.replace(_lm, '')
    text = re.sub(r'\\par\b', ' ', text)   # \par → space (\par is a word, \partial must survive)
    # ~ (LaTeX non-breaking space) → regular space
    # Lookahead [\w\\({] covers: Donald~Knuth, x~\alpha, a~(b+c), a~{expr}.
    # Lookbehind (?<=\w) ensures ~~ (Rich strikethrough) is never touched:
    # the first ~ in '~~' is preceded by non-word (start/space), so it doesn't match.
    text = re.sub(r'(?<=\w)~(?=[\w\\({])', ' ', text)

    # ── 2.8. \sqrt[n]{x} → ⁿ√(x)  /  \sqrt{x} → √(x) ────────────
    text = re.sub(
        r'\\sqrt\[(\d+)\]\{([^{}]*)\}',
        lambda m: m.group(1).translate(str.maketrans('0123456789', '⁰¹²³⁴⁵⁶⁷⁸⁹')) + '\u221a(' + m.group(2) + ')',
        text,
    )
    text = re.sub(r'\\sqrt\{([^{}]*)\}', '√(\\1)', text)

    # ── 3. Bare fractions ──────────────────────────────────────────
    # Normalise \dfrac / \tfrac / \cfrac → \frac first so _wrap_frac
    # finds them. These variants only differ in display size, which is
    # irrelevant in the terminal.
    text = text.replace(r'\dfrac', r'\frac')
    text = text.replace(r'\tfrac', r'\frac')
    text = text.replace(r'\cfrac', r'\frac')
    text = _wrap_frac_args_in_parens(text)

    # ── 4. Degree symbol ───────────────────────────────────────────────────
    # Primary fix is in _PRE_LATEX_MACRO_FIXUPS (converts ^\circ → ° before
    # pylatexenc sees it).  The following catch the bare-text path and any
    # leftovers (e.g., pylatexenc emits ∘ from \circ — catch that too).
    text = re.sub(r'(\d+)\\degree', r'\1°', text)    # 30\degree → 30°
    text = re.sub(r'(\d*)[\^]\{?\\circ\}?', r'\1°', text)  # 30^\circ / ^\circ
    # After pylatexenc: \circ may already be ∘ (U+2218); convert ^∘ → °
    text = re.sub(r'(\d*)\^\u2218', r'\1°', text)   # 30^∘ → 30° / ^∘ → °

    # ── 4.5. Accent / modifier macros ──────────────────────────────
    _HAT_MAP   = {'a':'\u00e2','e':'\u00ea','i':'\u00ee','o':'\u00f4','u':'\u00fb',
                  'A':'\u00c2','E':'\u00ca','I':'\u00ce','O':'\u00d4','U':'\u00db'}
    _TILDE_MAP = {'a':'\u00e3','n':'\u00f1','o':'\u00f5',
                  'A':'\u00c3','N':'\u00d1','O':'\u00d5'}
    _BAR_MAP   = {'a':'\u0101','e':'\u0113','i':'\u012b','o':'\u014d','u':'\u016b'}
    # ── Inner-first pass: process \vec/\dot/\ddot BEFORE \hat/\tilde/\bar ─────────
    # Reason: \hat{\vec{x}} — \vec{x} regex can't match through nested braces,
    # but \hat{ outer regex also fails (inner { breaks [^{}]+). Running \vec
    # first converts \vec{x}→x⃗, leaving \hat{x⃗} which \hat then catches.
    text = re.sub(r'\\vec\{([^{}]+)\}',
                  lambda m: m.group(1) + '\u20d7', text)
    text = re.sub(r'\\dot\{([^{}]+)\}',
                  lambda m: m.group(1) + '\u0307', text)
    text = re.sub(r'\\ddot\{([^{}]+)\}',
                  lambda m: m.group(1) + '\u0308', text)
    # ── Outer-pass: \hat/\tilde/\bar (run after inner macros expanded) ──────────
    # Pattern uses [^{}]+ (one-or-more) so multi-char args like \hat{AB} are
    # handled. For single chars the HAT/TILDE/BAR maps use precomposed Unicode;
    # the fallback is a combining diacritic on the full content.
    text = re.sub(r'\\hat\{([^{}]+)\}',
                  lambda m: (_HAT_MAP.get(m.group(1)) or m.group(1) + '\u0302'), text)
    text = re.sub(r'\\tilde\{([^{}]+)\}',
                  lambda m: (_TILDE_MAP.get(m.group(1)) or m.group(1) + '\u0303'), text)
    text = re.sub(r'\\bar\{([^{}]+)\}',
                  lambda m: (_BAR_MAP.get(m.group(1)) or m.group(1) + '\u0304'), text)
    text = re.sub(r'\\overline\{([^{}]+)\}',
                  lambda m: m.group(1) + '\u0305', text)
    text = re.sub(r'\\underline\{([^{}]+)\}', r'\1', text)
    text = re.sub(r'\\widehat\{([^{}]+)\}',
                  lambda m: m.group(1) + '\u0302', text)
    text = re.sub(r'\\widetilde\{([^{}]+)\}',
                  lambda m: m.group(1) + '\u0303', text)
    text = re.sub(r'\\overrightarrow\{([^{}]+)\}',
                  lambda m: m.group(1) + '\u20d7', text)
    text = re.sub(r'\\overleftarrow\{([^{}]+)\}',
                  lambda m: m.group(1) + '\u20d6', text)

    # ── 4.8. Ensure spaces around math functions BEFORE _bare ──────
    # Prevents fusing subscripts and functions (e.g. T_1\sin → T_1 sin),
    # which would cause bare subscript regex to mangle "1sin" into "₁ₛᵢₙ".
    text = re.sub(
        r'\\(sin|cos|tan|cot|sec|csc|log|ln|lim|max|min|det|deg|sinh|cosh|tanh|arcsin|arccos|arctan)\b',
        r' \1 ', text
    )

    # ── 5. Bare Greek + operators ──────────────────────────────────
    _bare = {
        # ── Greek lowercase — var-forms FIRST ───────────────────────
        r'\varepsilon': '\u03b5',  r'\epsilon': '\u03b5',
        r'\vartheta':   '\u03d1',  r'\theta':   '\u03b8',
        r'\varpi':      '\u03d6',  r'\pi':      '\u03c0',
        r'\varrho':     '\u03f1',  r'\rho':     '\u03c1',
        r'\varsigma':   '\u03c2',  r'\sigma':   '\u03c3',
        r'\varphi':     '\u03c6',  r'\phi':     '\u03c6',
        r'\alpha': '\u03b1',  r'\beta':  '\u03b2',  r'\gamma':  '\u03b3',
        r'\delta': '\u03b4',  r'\zeta':  '\u03b6',  r'\eta':    '\u03b7',
        r'\iota':  '\u03b9',  r'\kappa': '\u03ba',  r'\lambda': '\u03bb',
        r'\mu':    '\u03bc',  r'\nu':    '\u03bd',  r'\xi':     '\u03be',
        r'\tau':   '\u03c4',  r'\upsilon':'\u03c5', r'\chi':    '\u03c7',
        r'\psi':   '\u03c8',  r'\omega': '\u03c9',
        # ── Greek uppercase ──────────────────────────────────────────
        r'\Gamma': '\u0393',  r'\Delta': '\u0394',  r'\Theta':  '\u0398',
        r'\Lambda':'\u039b',  r'\Xi':    '\u039e',  r'\Pi':     '\u03a0',
        r'\Sigma': '\u03a3',  r'\Upsilon':'\u03a5', r'\Phi':    '\u03a6',
        r'\Psi':   '\u03a8',  r'\Omega': '\u03a9',
        # ── Relations: longer/more-specific aliases FIRST ───────────
        # ORDERING: \leq BEFORE \le, \geq BEFORE \ge because \le is a
        # prefix of \leq — if \le replaced first, \leq → ≤q (garbled).
        r'\models':  '\u22a8',  # \models BEFORE \bmod/\mod (\mod is prefix of \models)
        r'\bmod':  ' mod ',
        r'\mod':   ' mod ',
        r'\degree':'°',
        # \lesssim/\gtrsim BEFORE \leq/\le (\le is prefix of \lesssim — would eat 's')
        r'\lesssim':  '\u2272',  r'\gtrsim':  '\u2273',
        r'\leq':   '\u2264',  r'\geq':   '\u2265',
        r'\le':    '\u2264',  r'\ge':    '\u2265',
        r'\times': '\u00d7',
        r'\cdots':  '\u22ef',
        r'\cdot':  '\u00b7',
        r'\pm':    '\u00b1',  r'\mp':    '\u2213',
        r'\div':   '\u00f7',  r'\ast':   '\u2217',
        r'\implies': ' \u21d2 ',   r'\iff':      ' \u21d4 ',
        r'\therefore':' \u2234 ',  r'\because':  ' \u2235 ',
        r'\lVert':   '\u2016',     r'\rVert':    '\u2016',   r'\Vert': '\u2016',
        r'\lvert':   '|',          r'\rvert':    '|',
        r'\supseteq': '\u2287',  r'\subseteq': '\u2286',
        r'\supset':   '\u2283',  r'\subset':   '\u2282',
        r'\approx':   '\u2248',  r'\neq':      '\u2260',   r'\equiv': '\u2261',
        r'\gg':       '\u226b',  r'\ll':       '\u226a',
        r'\propto':   '\u221d',
        r'\simeq':    '\u2243',  r'\sim':      '\u223c',
        r'\cong':     '\u2245',  # ≅ — must follow \simeq/\sim (no prefix clash)
        # ── Negated relations — MUST precede their positive counterparts ──────
        # e.g. \nleq before \leq, \nless before \le, \ngeq before \geq/\ge
        r'\nless':    '\u226e',  r'\ngtr':    '\u226f',
        r'\nleq':     '\u2270',  r'\ngeq':    '\u2271',
        # ── Partial orders ──────────────────────────────────────────
        # \preceq BEFORE \prec  |  \succeq BEFORE \succ
        r'\preceq':   '\u2aaf',  r'\succeq':  '\u2ab0',
        r'\prec':     '\u227a',  r'\succ':    '\u227b',
        # ── Approximation with tilde ──────────────────────────────
        r'\infty':  '\u221e',  r'\partial': '\u2202',  r'\nabla': '\u2207',
        r'\iiint':  '\u222d',  r'\iint':    '\u222c',  r'\oint':  '\u222e',
        r'\inf':     'inf',     r'\sup':     'sup',
        r'\int':    '\u222b',
        r'\notin':    '\u2209',  r'\in':       '\u2208',
        r'\nmid':     '\u2224',  r'\mid':      '\u2223',
        r'\sum':    '\u2211',  r'\prod':    '\u220f',  r'\coprod': '\u2210',
        r'\sqrt':   '\u221a',
        r'\cdots':  '\u22ef',  r'\ldots':   '\u2026',  r'\vdots': '\u22ee',
        r'\ddots':  '\u22f1',  r'\dots':    '\u2026',
        # ── Long arrows — ALL before their short counterparts ───────────────
        # \longrightarrow is a superset of \rightarrow as a raw string.
        # If \rightarrow ran first, \longrightarrow → \long→ (garbled).
        r'\Longleftrightarrow': '\u21d4',  # long double ⇔
        r'\longleftrightarrow': '\u2194',  # long single ↔
        r'\Longrightarrow':     '\u21d2',  # long double ⇒
        r'\Longleftarrow':      '\u21d0',  # long double ⇐
        r'\longrightarrow':     '\u2192',  # long single →
        r'\longleftarrow':      '\u2190',  # long single ←
        r'\rightleftharpoons': '\u21cc',  r'\rightleftarrows': '\u21c4',
        r'\leftrightarrow':    '\u2194',  r'\Leftrightarrow':  '\u21d4',
        r'\rightarrow':        '\u2192',  r'\leftarrow':       '\u2190',
        r'\Rightarrow':        '\u21d2',  r'\Leftarrow':       '\u21d0',
        r'\uparrow':   '\u2191',  r'\downarrow':  '\u2193',
        r'\updownarrow':'\u2195',
        r'\nearrow':   '\u2197',  r'\searrow':    '\u2198',
        r'\swarrow':   '\u2199',  r'\nwarrow':    '\u2196',
        r'\mapsto':    '\u21a6',
        r'\dagger':   '\u2020', r'\ddagger':  '\u2021',
        r'\coloneqq': '\u2254', r'\coloneq':  '\u2254',  # := notation
        r'\top':      '\u22a4', r'\bot':      '\u22a5',
        r'\to':       '\u2192',
        # ── Set theory ──────────────────────────────────────────
        r'\cup':      '\u222a',  r'\cap':     '\u2229',
        r'\setminus': '\u2216',  # A ∖ B set difference
        r'\emptyset': '\u2205',  r'\varnothing':'\u2205',
        r'\complement':'\u2201',  # A^\complement
        r'\nexists': '\u2204',  r'\exists': '\u2203',
        r'\land':    '\u2227',  r'\lor':    '\u2228',
        r'\wedge':   '\u2227',  r'\vee':    '\u2228',   # aliases for \land/\lor
        r'\lnot':    '\u00ac',  r'\neg':    '\u00ac',
        r'\forall':  '\u2200',
        # ── Algebra / operators ───────────────────────────────────────
        r'\oplus':   '\u2295',  r'\otimes':  '\u2297',
        r'\odot':    '\u2299',  r'\ominus':  '\u2296',
        r'\circ':    '\u2218',  r'\bullet':  '\u2022',
        # \models moved above before \bmod/\mod (see ordering comment there)
        r'\vdash':  '\u22a2',  r'\dashv': '\u22a3',   # proof turnstile / reverse turnstile
        # ── Brackets ────────────────────────────────────────────────
        r'\lceil':   '\u2308',  r'\rceil':   '\u2309',   # ⌈ ⌉ ceiling
        r'\lfloor':  '\u230a',  r'\rfloor':  '\u230b',   # ⌊ ⌋ floor
        # ── Colon forms ────────────────────────────────────────────────
        r'\colon':   ':',  # f \colon X \to Y — function type notation
        r'\mathbb{R}': '\u211d',  r'\mathbb{N}': '\u2115',
        r'\mathbb{Z}': '\u2124',  r'\mathbb{Q}': '\u211a',
        r'\mathbb{C}': '\u2102',  r'\mathbb{H}': '\u210d',
        r'\mathbb{P}': '\u2119',  r'\mathbb{E}': '\U0001d53c',
        r'\prime':    '\u2032',  r'\angle':    '\u2220',
        r'\perp':     '\u22a5',  r'\parallel': '\u2225',
        r'\hbar':     '\u210f',  r'\ell':      '\u2113',
        r'\imath':    '\u0131',  r'\jmath':    '\u0237',  # dotless i/j for vectors
        r'\triangle': '\u25b3',  r'\square':   '\u25a1',
        r'\diamond':  '\u25c7',
        r'\arcsin':  'arcsin',  r'\arccos':  'arccos',  r'\arctan':  'arctan',
        r'\arccot':  'arccot',  r'\arcsec':  'arcsec',  r'\arccsc':  'arccsc',
        r'\sinh':    'sinh',    r'\cosh':    'cosh',
        r'\tanh':    'tanh',    r'\coth':    'coth',
        r'\sin':     'sin',     r'\cos':     'cos',     r'\tan':     'tan',
        r'\cot':     'cot',     r'\sec':     'sec',     r'\csc':     'csc',
        r'\lim':     'lim',     r'\max':     'max',     r'\min':     'min',
        # (moved to relations block)
        r'\det':     'det',     r'\log':     'log',     r'\ln':      'ln',
        r'\exp':     'exp',     r'\gcd':     'gcd',     r'\lcm':     'lcm',
        r'\Pr':      'Pr',      r'\Re':      'Re',      r'\Im':      'Im',
        r'\ker':     'ker',     r'\dim':     'dim',     r'\deg':     'deg',
        r'\arg':     'arg',     r'\hom':     'Hom',
        r'\tr':      'tr',      r'\rank':    'rank',
    }
    # ── 5-pre. Regex-based macros (BEFORE _bare for brace-args; AFTER for \mathbb) ──
    # \pmod{n}, \pod{n}, \bmod{n}: take brace args .replace() can't extract.
    text = re.sub(r'\\pmod\{([^{}]*)\}', r' (mod \1)', text)
    text = re.sub(r'\\pod\{([^{}]*)\}',  r' (\1)',     text)
    text = re.sub(r'\\bmod\{([^{}]*)\}', r' mod \1',  text)
    # \binom and \boxed: same treatment as in _apply_pre_latex_fixups.
    text = re.sub(r'\\[dt]?binom\{([^{}]*)\}\{([^{}]*)\}', r'C(\1,\2)', text)
    text = re.sub(r'\\boxed\{([^{}]*)\}', '\u27e6' + r'\1' + '\u27e7', text)
    # \multicolumn{n}{align}{content}: strip the column/align args, keep content.
    # Must run BEFORE _bare so \mu doesn't eat the 'lticolumn' prefix.
    text = re.sub(r'\\multicolumn\{[^{}]*\}\{[^{}]*\}\{([^{}]*)\}', r'\1', text)

    # Bug #43 fix: sort by descending key length so longer (more-specific)
    # macros like \infty always replace before their shorter prefixes like
    # \inf, and \notin replaces before \in.  Dict insertion order is
    # maintained in Python 3.7+ but doesn't guarantee longest-first.
    for latex, uni in sorted(_bare.items(), key=lambda kv: len(kv[0]), reverse=True):
        text = text.replace(latex, uni)

    # ── 5-post. \mathbb catch-all (AFTER _bare so R→ℝ etc. fire first) ───────────
    # Anything remaining after _bare's specific entries is an unlisted field
    # letter — fall back to stripping the wrapper (bare letter is readable).
    text = re.sub(r'\\mathbb\{([^{}]*)\}', r'\1', text)

    # ── 6. Bare superscripts/subscripts ────────────────────────────
    sup_map = str.maketrans(
        '0123456789+-=()abcdefghijklmnopqrstuvwxyz',
        '\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079'
        '\u207a\u207b\u207c\u207d\u207e'
        '\u1d43\u1d47\u1d9c\u1d48\u1d49\u1da0\u1d4d\u02b0\u2071\u02b2'
        '\u1d4f\u02e1\u1d50\u207f\u1d52\u1d56q\u02b3\u02e2\u1d57\u1d58'
        '\u1d5b\u02b7\u02e3\u02b8\u1dbb'
    )
    sub_map = str.maketrans(
        '0123456789aehijklmnoprstuvx',
        '₀₁₂₃₄₅₆₇₈₉ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ'
    )
    _sub_chars = set('0123456789aehijklmnoprstuvx')

    # ── 6a. Compound fraction exponents (bare path, outside math blocks) ────
    text = re.sub(r'\^\{(\d+)/(\d+)\}', _fmt_frac_exp, text)

    # ── 6a2. Prime superscripts — BEFORE general ^{...} ─────────────────────
    # f^{\prime\prime\prime} → f‴, f^{\prime\prime} → f″, f^{\prime} → f′
    # Also handle the common apostrophe-style f'' → f″ etc.
    text = re.sub(r"\^\{(?:\\prime){3}\}", "\u2034", text)   # ‴ triple prime (raw)
    text = re.sub(r"\^\{(?:\\prime){2}\}", "\u2033", text)   # ″ double prime (raw)
    text = re.sub(r"\^\{\\prime\}",        "\u2032", text)   # ′ single prime (raw)
    # Also match already-converted form — _bare converts \prime → ′ BEFORE us,
    # so ^{\prime\prime} → ^{′′} by the time this section runs.
    text = re.sub(r"\^\{[\u2032]{3}\}", "\u2034", text)      # ‴ triple prime (already converted)
    text = re.sub(r"\^\{[\u2032]{2}\}", "\u2033", text)      # ″ double prime (already converted)
    text = re.sub(r"\^\{[\u2032]\}",    "\u2032", text)      # ′ single prime (already converted)

    # ── 6b. Regular superscripts ──────────────────────────────────
    text = re.sub(
        r'\^\{([0-9+\-=()a-zA-Z]+)\}|\^([+-]?\d+)',
        lambda m: (m.group(1) or m.group(2)).translate(sup_map),
        text,
    )
    # ── 6c. Fallback: ^{...} that survived ────────────────────────
    text = re.sub(r'\^\{([^{}]+)\}', lambda m: '^(' + m.group(1) + ')', text)

    def _sub_braced(m: re.Match) -> str:
        content = m.group(1)
        if all(c in _sub_chars for c in content):
            return content.translate(sub_map)
        return f"_{content}"

    text = re.sub(r'_\{(\w+)\}', _sub_braced, text)
    text = re.sub(r'_([0-9aehijklmnoprstuvx]+)(?![a-zA-Z0-9_])',
        lambda m: m.group(1).translate(sub_map), text)

    # ── 7. Restore stashed inline code spans ──────────────────────
    for i, original in enumerate(_stash):
        text = text.replace(_PLACEHOLDER.format(i), original)

    # ── 8. Final whitespace normalisation ───────────────────────────
    text = re.sub(r'[ \t]{2,}', ' ', text)

    return text


# ══════════════════════════════════════════════════════════════════════════════
# RESPONSE PRINTER
# ══════════════════════════════════════════════════════════════════════════════

def print_smart_response(text: str):
    """Print response — syntax-highlighted code blocks, markdown for prose with LaTeX rendering."""
    try:
        parts = re.split(r"(```[\s\S]*?```)", text)
        for part in parts:
            if not part.strip():
                continue
            if part.strip().startswith("```"):
                lines = part.strip().splitlines()
                language = lines[0].replace("```", "").strip() or "text"
                code = "\n".join(lines[1:-1])
                console.print(Syntax(code, language, theme="catppuccin-mocha",
                                     line_numbers=False, word_wrap=True))
            else:
                console.print(Markdown(prep_for_console(part.strip())))
    except Exception:
        console.print(text)