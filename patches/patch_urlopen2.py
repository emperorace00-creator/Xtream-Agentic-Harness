import sys

with open('image_ocr_agent.py', encoding='utf-8') as f:
    src = f.read()

old = (
    "                    import urllib.request as _urllib_req\n"
    "                    with _urllib_req.urlopen(image_url, timeout=15) as _r:\n"
    "                        _raw = _r.read()\n"
    "                    parts.append(types.Part.from_bytes(\n"
    "                        data=_raw, mime_type=mime_type_from_url(image_url)\n"
    "                    ))\n"
    "                except Exception as e:\n"
    '                    raise RuntimeError(f"Could not download image URL for Google OCR multi: {e}")'
)

new = (
    "                    import urllib.request as _urllib_req\n"
    "                    # Bug #16 + #15 fix: browser User-Agent to bypass CDN 403s;\n"
    "                    # 20 MB read cap to prevent OOM on unexpectedly large images.\n"
    "                    _req = _urllib_req.Request(\n"
    "                        image_url,\n"
    '                        headers={"User-Agent": "Mozilla/5.0 (compatible; ImageOCR/1.0)"},\n'
    "                    )\n"
    "                    with _urllib_req.urlopen(_req, timeout=15) as _r:\n"
    "                        _raw = _r.read(20 * 1024 * 1024 + 1)\n"
    "                    if len(_raw) > 20 * 1024 * 1024:\n"
    '                        raise ValueError("Remote image exceeds 20 MB limit")\n'
    "                    parts.append(types.Part.from_bytes(\n"
    "                        data=_raw, mime_type=mime_type_from_url(image_url)\n"
    "                    ))\n"
    "                except Exception as e:\n"
    '                    raise RuntimeError(f"Could not download image URL for Google OCR multi: {e}")'
)

if old not in src:
    print("NOT FOUND — pattern missing")
    sys.exit(1)

src = src.replace(old, new, 1)
with open('image_ocr_agent.py', 'w', encoding='utf-8') as f:
    f.write(src)
print("PATCHED OK")
