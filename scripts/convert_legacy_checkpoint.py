"""Wrap a bare ``state_dict`` checkpoint into the self-describing format.

The original training scripts saved weights with::

    torch.save(model.state_dict(), path)

which records the tensors and nothing else: not the architecture that produced
them, not the preprocessing the inputs went through, not the decision threshold
the results were computed at. Months later there is no way to tell from the file
whether ``base_channels`` was 32 or 64, and loading it into the wrong
architecture either raises a confusing key error or, worse, succeeds.

This script wraps such a file into the format :mod:`msseg.evaluate` expects, so
an existing checkpoint can be evaluated without retraining. It validates that
the weights actually fit the declared architecture before writing anything.

Usage::

    python scripts/convert_legacy_checkpoint.py \\
        --input  "path/to/unet2d_nature_multimodal_best.pth" \\
        --output runs/thesis_reproduction/best.pt \\
        --processed-root data/processed_legacy \\
        --manifest data/manifests/splits_legacy.csv \\
        --threshold 0.7

Then::

    python -m msseg.evaluate --checkpoint runs/thesis_reproduction/best.pt \\
        --split test --output-dir results/thesis_reproduction
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from msseg.config import DataConfig, EvalConfig, ExperimentConfig, ModelConfig, TrainConfig
from msseg.models.unet import UNet2D

logger = logging.getLogger(__name__)


def convert(
    input_path: Path,
    output_path: Path,
    processed_root: str,
    manifest: str,
    threshold: float,
    base_channels: int,
    in_channels: int,
    use_se: bool,
    mode: str,
    min_component_size: int,
) -> None:
    """Validate a bare state_dict against the architecture and wrap it."""
    payload = torch.load(input_path, map_location="cpu", weights_only=False)

    if isinstance(payload, dict) and "model_state_dict" in payload:
        raise SystemExit(
            f"{input_path} is already in the new format. Nothing to convert."
        )
    if not isinstance(payload, dict):
        raise SystemExit(
            f"{input_path} does not contain a state_dict "
            f"(found {type(payload).__name__}). Was a whole model pickled?"
        )

    state_dict = payload
    model = UNet2D(
        in_channels=in_channels,
        out_channels=1,
        base_channels=base_channels,
        use_se=use_se,
    )

    # Strict load: refuses to write a checkpoint whose weights do not actually
    # belong to the architecture being recorded alongside them.
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        reference = model.state_dict()
        missing = sorted(set(reference) - set(state_dict))
        unexpected = sorted(set(state_dict) - set(reference))
        mismatched = [
            f"{k}: file has {tuple(state_dict[k].shape)}, "
            f"architecture expects {tuple(reference[k].shape)}"
            for k in reference
            if k in state_dict and state_dict[k].shape != reference[k].shape
        ]

        lines = ["The weights do not fit the declared architecture."]
        if missing:
            lines.append(f"  missing from the file  : {missing[:6]}")
        if unexpected:
            lines.append(f"  unexpected in the file : {unexpected[:6]}")
        if mismatched:
            lines.append("  shape mismatches:")
            lines += [f"    {m}" for m in mismatched[:6]]
            lines.append(
                "  A mismatch on enc1.block.0.weight means --in-channels or "
                "--base-channels is wrong."
            )
        lines.append("Check --base-channels, --in-channels and --use-se / --no-se.")
        raise SystemExit("\n".join(lines)) from error

    cfg = ExperimentConfig(
        name="thesis_reproduction",
        data=DataConfig(processed_root=processed_root, manifest=manifest, mode=mode),
        model=ModelConfig(
            in_channels=in_channels,
            out_channels=1,
            base_channels=base_channels,
            use_se=use_se,
        ),
        train=TrainConfig(selection_metric="soft_dice", fixed_threshold=threshold),
        eval=EvalConfig(threshold=threshold, min_component_size=min_component_size),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": state_dict,
            "config": cfg.to_dict(),
            "epoch": -1,
            "val_score": float("nan"),
            "threshold": threshold,
            "converted_from": str(input_path),
        },
        output_path,
    )

    logger.info("Converted %s -> %s", input_path, output_path)
    logger.info("  %d tensors, %.2fM parameters", len(state_dict), model.num_parameters / 1e6)
    logger.info("  threshold %.2f, min component size %d", threshold, min_component_size)
    logger.warning(
        "The weights were trained on a specific intensity normalisation. "
        "Evaluate them on data preprocessed the same way (legacy_minmax for the "
        "original thesis scripts), or the metrics will be low with no error."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", type=Path, required=True, help="The .pth to convert.")
    parser.add_argument("--output", type=Path, required=True, help="Destination .pt.")
    parser.add_argument("--processed-root", default="data/processed_legacy")
    parser.add_argument("--manifest", default="data/manifests/splits_legacy.csv")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--min-component-size", type=int, default=10)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--mode", default="multimodal", choices=("multimodal", "axial_context"))
    se = parser.add_mutually_exclusive_group()
    se.add_argument("--use-se", dest="use_se", action="store_true", default=True)
    se.add_argument("--no-se", dest="use_se", action="store_false")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    convert(
        input_path=args.input,
        output_path=args.output,
        processed_root=args.processed_root,
        manifest=args.manifest,
        threshold=args.threshold,
        base_channels=args.base_channels,
        in_channels=args.in_channels,
        use_se=args.use_se,
        mode=args.mode,
        min_component_size=args.min_component_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
