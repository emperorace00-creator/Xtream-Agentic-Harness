"""
apply_all_patches.py — Run all patch batches in sequence.

Usage (from project root):
    python patches/apply_all_patches.py

Each batch prints PATCH/SKIP/FAIL per fix.
SKIP = already applied (or pattern changed — verify manually).
FAIL = ambiguous match (multiple occurrences) — needs manual review.
"""

import subprocess
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATCHES_DIR = os.path.dirname(os.path.abspath(__file__))

BATCHES = [
    ("patch_batch1_security_crashes.py",
     "Batch 1 — Security, Crashes & Tool Failures"),
    ("patch_batch2_state_integrity.py",
     "Batch 2 — State & Data Integrity"),
    ("patch_batch3_renderer_ocr.py",
     "Batch 3 — Renderer, OCR & Duplicate Injection"),
]


def main():
    print("=" * 70)
    print("Applying all bug-fix patches")
    print("=" * 70)
    print()

    failed = []
    for script, label in BATCHES:
        script_path = os.path.join(PATCHES_DIR, script)
        if not os.path.isfile(script_path):
            print(f"[MISSING] {script} — skipping")
            failed.append(script)
            continue

        print(f">>> {label}")
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=ROOT,
        )
        if result.returncode != 0:
            print(f"[ERROR] {script} exited with code {result.returncode}")
            failed.append(script)
        print()

    print("=" * 70)
    if failed:
        print(f"DONE with {len(failed)} failure(s): {', '.join(failed)}")
        print("Review FAIL/SKIP output above and apply those manually.")
        sys.exit(1)
    else:
        print("All patches applied successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
