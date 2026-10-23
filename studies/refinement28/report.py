"""Read completed cases and create 24 original-size 5/28 versus 8/28 panels."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import os
from pathlib import Path
import re
import tempfile

from .common import OUT, complete, folder, freeze_check, read, sha, write


def atomic_bytes(path, data):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def completion_rows(manifest):
    rows = []
    for row in manifest["rows"]:
        directory = folder(row)
        status_path = directory / "status.json"
        try:
            status = read(status_path) if status_path.exists() else {}
            if not isinstance(status, dict):
                status = {"state": "unreadable_status"}
        except (OSError, ValueError):
            status = {"state": "unreadable_status"}
        passed = complete(row, verify=True)
        state = "complete" if passed else status.get("state", "missing")
        if not passed and state == "complete":
            state = "invalid_completed_artifacts"
        result = dict(index=row["index"], case_id=row["case_id"], prompt=row["prompt"],
                      refinement_steps=row["refinement_steps"], state=state, verified_complete=passed,
                      image_path=str(directory / "final.png"), image_sha256=None,
                      generation_seconds=None, peak_allocated_gib=None)
        if passed:
            metadata = read(directory / "metadata.json")
            result.update(image_sha256=sha(directory / "final.png"),
                          generation_seconds=metadata.get("generation_seconds"),
                          peak_allocated_gib=metadata.get("peak_allocated_gib"))
        rows.append(result)
    return rows


def make_panel(left, right, target):
    """Paste the two ERPs at their stored dimensions; no resampling or repairs."""
    from PIL import Image, ImageDraw, ImageFont
    with Image.open(left["image_path"]) as first, Image.open(right["image_path"]) as second:
        first.load()
        second.load()
        if first.mode != "RGB" or second.mode != "RGB" or first.size != second.size:
            raise ValueError("Comparison ERPs must have identical RGB dimensions")
        width, height = first.size
        if (width, height) != (4096, 2048):
            raise ValueError("Expected preserved 4096 x 2048 ERP outputs")
        pad, header = 24, 112
        canvas = Image.new("RGB", (width * 2 + pad * 3, height + header + pad), "white")
        draw = ImageDraw.Draw(canvas)
        try:
            title_font = ImageFont.truetype("DejaVuSans.ttf", 30)
            font = ImageFont.truetype("DejaVuSans.ttf", 26)
        except OSError:
            title_font = font = ImageFont.load_default()
        draw.text((pad, 12), left["prompt"], fill="black", font=title_font)
        draw.text((pad, 60), "poisson_select + screened | refinement 5/28", fill="black", font=font)
        draw.text((width + 2 * pad, 60), "poisson_select + screened | refinement 8/28", fill="black", font=font)
        canvas.paste(first, (pad, header))
        canvas.paste(second, (width + 2 * pad, header))
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    atomic_bytes(target, buffer.getvalue())
    return dict(path=str(target), width=canvas.width, height=canvas.height,
                image_dimensions=[width, height], interpolation="none", prompt=left["prompt"],
                left_image_sha256=left["image_sha256"], right_image_sha256=right["image_sha256"])


def generate_report():
    manifest = freeze_check()
    with (OUT / "report.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = completion_rows(manifest)
        grouped = {}
        for row in rows:
            variants = grouped.setdefault(row["prompt"], {})
            if row["refinement_steps"] in variants:
                raise ValueError("Duplicate refinement case for a prompt")
            variants[row["refinement_steps"]] = row
        target = OUT / "figures"
        target.mkdir(exist_ok=True)
        panels = []
        for prompt, variants in grouped.items():
            if set(variants) != {5, 8} or not all(row["verified_complete"] for row in variants.values()):
                continue
            stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(prompt).stem)
            suffix = hashlib.sha256(prompt.encode()).hexdigest()[:8]
            panels.append(make_panel(variants[5], variants[8], target / (stem + "-" + suffix + "-refinement5-vs8.png")))
        buffer = io.StringIO(newline="")
        fields = ["index", "case_id", "prompt", "refinement_steps", "state", "verified_complete", "image_path",
                  "image_sha256", "generation_seconds", "peak_allocated_gib"]
        writer = csv.DictWriter(buffer, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        atomic_bytes(OUT / "completion.csv", buffer.getvalue().encode())
        completed = sum(row["verified_complete"] for row in rows)
        summary = dict(state="complete" if completed == len(rows) else "incomplete", expected_cases=len(rows),
                       verified_complete_cases=completed, incomplete_cases=len(rows) - completed,
                       expected_pairs=len(grouped), complete_pairs=len(panels), cases=rows, figures=panels,
                       source_sha256=manifest["source_sha256"], manifest_sha256=sha(OUT / "manifest.json"),
                       created_utc=datetime.now(timezone.utc).isoformat(),
                       note="Completion and original-size ERP comparison only; no GPU evaluation, metric calculation, or historical artifact changes.")
        write(OUT / "completion.json", summary)
        write(target / "index.json", panels)
        return summary


def main():
    import json
    result = generate_report()
    print(json.dumps({key: value for key, value in result.items() if key not in ("cases", "figures")}, indent=2))


if __name__ == "__main__":
    main()
