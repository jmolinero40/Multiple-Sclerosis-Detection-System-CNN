"""Loss functions for heavily imbalanced binary segmentation.

MS lesions occupy roughly 0.1-1% of the pixels in an axial FLAIR slice. A plain
binary cross-entropy converges to predicting "background everywhere", which
scores >99% pixel accuracy and a Dice of 0. Two mechanisms counteract this:

* a **soft Dice** term, which is scale-invariant with respect to the size of the
  foreground and therefore does not vanish for small lesions;
* a **positive class weight** inside the BCE term.

``BCEDiceLoss`` combines both. The default ``bce_weight=0.7`` was selected on the
validation split.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["soft_dice", "DiceLoss", "BCEDiceLoss"]


def soft_dice(probs: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Mean soft Dice coefficient over the batch.

    Args:
        probs: Predicted probabilities in ``[0, 1]``, shape ``(B, 1, H, W)``.
        target: Binary ground truth in ``{0, 1}``, same shape.
        eps: Smoothing constant. Also defines the value for an empty
            prediction on an empty mask, which becomes ``1.0``.

    Returns:
        Scalar tensor: the Dice coefficient averaged over the batch.
    """
    b = probs.shape[0]
    probs = probs.contiguous().view(b, -1)
    target = target.contiguous().view(b, -1)

    intersection = (probs * target).sum(dim=1)
    denominator = probs.sum(dim=1) + target.sum(dim=1)

    dice = (2.0 * intersection + eps) / (denominator + eps)
    return dice.mean()


class DiceLoss(nn.Module):
    """``1 - soft_dice``. Expects logits."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1.0 - soft_dice(torch.sigmoid(logits), target, eps=self.eps)


class BCEDiceLoss(nn.Module):
    """Convex combination of weighted BCE and soft Dice loss.

    ``loss = bce_weight * BCE + (1 - bce_weight) * (1 - soft_dice)``

    Args:
        bce_weight: Weight of the BCE term, in ``[0, 1]``.
        pos_weight: Weight applied to the positive class inside BCE. A value
            above 1 trades precision for recall. Passed straight to
            :class:`torch.nn.BCEWithLogitsLoss`, so it must be a tensor on the
            same device as the logits, or ``None``.

    Note:
        Uses ``BCEWithLogitsLoss`` (log-sum-exp trick) rather than
        ``sigmoid`` + ``BCELoss``, which is numerically unstable for saturated
        logits.
    """

    def __init__(self, bce_weight: float = 0.7, pos_weight: torch.Tensor | None = None) -> None:
        super().__init__()
        if not 0.0 <= bce_weight <= 1.0:
            raise ValueError(f"bce_weight must be in [0, 1], got {bce_weight}")
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_term = self.bce(logits, target)
        dice_term = 1.0 - soft_dice(torch.sigmoid(logits), target)
        return self.bce_weight * bce_term + (1.0 - self.bce_weight) * dice_term
