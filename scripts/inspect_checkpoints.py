"""Identify what architecture each checkpoint in a folder actually holds.

A folder of `.pth` files accumulated over months of experiments is unreadable:
the filenames record intentions, not contents, and a bare `state_dict` carries
no record of the architecture that produced it. This script reads the tensor
shapes and reconstructs the architecture from them.

What it can tell from the weights alone:

* ``in_channels``  -- second dimension of the first convolution
* ``base_channels``-- first dimension of the same tensor
* SE blocks        -- presence of any ``*.se.fc.*`` parameter
* normalisation    -- BatchNorm keeps ``running_mean`` buffers, GroupNorm does not
* depth            -- how many encoder levels exist

What it cannot tell, because weights do not record it: which dataset the model
was trained on, which preprocessing the inputs went through, or what decision
threshold the reported numbers used. Those live in the config, which is why
:mod:`msseg.train` now writes the config into every checkpoint it saves.

Usage::

    python scripts/inspect_checkpoints.py --folder "C:/Users/jmoli/Desktop/TFG mates/Modelos"
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import torch

# Known configurations in this project, keyed by (in_channels, base_channels, has_se).
KNOWN = {
    (3, 32, True): "3-channel + SE  (multimodal FLAIR/T1/T2, or 2.5D z-1/z/z+1)",
    (3, 32, False): "3-channel, no SE  (the SE ablation)",
    (1, 32, True): "1-channel + SE  (single FLAIR slice)",
    (1, 32, False): "1-channel, no SE",
}


def describe(path: Path) -> dict:
    """Read one checkpoint and infer what it is."""
    row: dict = {"file": path.name, "size_mb": path.stat().st_size / 1e6,
                 "modified": dt.datetime.fromtimestamp(path.stat().st_mtime)}

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        row["note"] = f"could not load: {type(error).__name__}"
        return row

    # New-format checkpoints carry their own config; nothing to infer.
    if isinstance(payload, dict) and "model_state_dict" in payload:
        cfg = payload.get("config", {})
        model_cfg = cfg.get("model", {})
        row.update(
            format="self-describing",
            in_channels=model_cfg.get("in_channels"),
            base_channels=model_cfg.get("base_channels"),
            has_se=model_cfg.get("use_se"),
            threshold=payload.get("threshold"),
            note=f"config embedded: {cfg.get('name', '?')}",
        )
        return row

    if not isinstance(payload, dict):
        row["note"] = f"not a state_dict ({type(payload).__name__})"
        return row

    state = payload
    row["format"] = "bare state_dict"
    row["tensors"] = len(state)
    row["params"] = sum(v.numel() for v in state.values() if hasattr(v, "numel"))

    first = state.get("enc1.block.0.weight")
    if first is None:
        row["note"] = "not a UNet2D from this project"
        return row

    row["base_channels"] = int(first.shape[0])
    row["in_channels"] = int(first.shape[1])
    row["has_se"] = any(".se.fc." in k for k in state)
    row["norm"] = "BatchNorm" if any("running_mean" in k for k in state) else "GroupNorm"
    row["levels"] = sum(1 for k in state if k.endswith("enc1.block.0.weight") or
                        (k.startswith("enc") and k.endswith(".block.0.weight")))
    row["note"] = KNOWN.get(
        (row["in_channels"], row["base_channels"], row["has_se"]), "unrecognised combination"
    )
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--sort", choices=("name", "date"), default="date")
    args = parser.parse_args(argv)

    files = sorted(
        [p for p in args.folder.iterdir() if p.suffix.lower() in (".pth", ".pt")],
        key=(lambda p: p.stat().st_mtime) if args.sort == "date" else (lambda p: p.name.lower()),
    )
    if not files:
        print(f"No .pth or .pt files under {args.folder}")
        return 1

    rows = [describe(p) for p in files]

    width = max(len(r["file"]) for r in rows)
    header = (f"{'file':<{width}}  {'modified':<16}  {'MB':>6}  {'in':>3}  "
              f"{'base':>4}  {'SE':>3}  {'norm':<9}  what it is")
    print(header)
    print("-" * len(header))

    for r in rows:
        se = "-" if r.get("has_se") is None else ("yes" if r["has_se"] else "no")
        print(
            f"{r['file']:<{width}}  {r['modified']:%Y-%m-%d %H:%M}  {r['size_mb']:>6.1f}  "
            f"{str(r.get('in_channels', '?')):>3}  {str(r.get('base_channels', '?')):>4}  "
            f"{se:>3}  {r.get('norm', '-'):<9}  {r.get('note', '')}"
        )

    print(
        "\nArchitecture is read from the weights and is reliable. Which dataset, "
        "preprocessing and threshold each one used is NOT recorded in a bare "
        "state_dict -- match those against the script that saved it."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
