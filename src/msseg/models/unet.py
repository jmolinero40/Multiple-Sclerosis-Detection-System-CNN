"""2D U-Net with Squeeze-and-Excitation blocks for MS lesion segmentation.

The architecture is a standard 4-level U-Net (Ronneberger et al., 2015) with two
modifications that matter for this task:

1. **Squeeze-and-Excitation** (Hu et al., 2018) after every double-convolution
   block. SE learns a per-channel gate, which lets the network weight the three
   input modalities (FLAIR / T1 / T2) differently depending on the content of
   the slice. This is the main reason SE was chosen over spatial attention:
   the modality axis *is* the channel axis at the input.

2. **GroupNorm instead of BatchNorm.** Training runs with batch size 8 on a
   single consumer GPU. BatchNorm statistics are noisy at that batch size,
   and the foreground (lesion) occupies well under 1% of the pixels, so the
   running statistics are dominated by background. GroupNorm is batch-independent
   and was empirically more stable here.

The network outputs raw logits; apply ``torch.sigmoid`` to obtain probabilities.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["SEBlock", "DoubleConv", "UNet2D"]


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention.

    Squeezes each channel to a scalar via global average pooling, learns a
    gate in ``[0, 1]`` through a bottleneck MLP, and rescales the channel.

    Args:
        channels: Number of input (and output) channels.
        reduction: Bottleneck ratio. ``hidden = max(channels // reduction, 1)``.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.fc(self.pool(x))  # (B, C, 1, 1)
        return x * weights


class DoubleConv(nn.Module):
    """(Conv3x3 -> GroupNorm -> ReLU) x 2, optionally followed by an SE block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_se: bool = True,
        se_reduction: int = 16,
        gn_groups: int = 8,
    ) -> None:
        super().__init__()
        if out_channels % gn_groups != 0:
            raise ValueError(
                f"out_channels={out_channels} must be divisible by gn_groups={gn_groups}"
            )
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=gn_groups, num_channels=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=gn_groups, num_channels=out_channels),
            nn.ReLU(inplace=True),
        )
        self.se = SEBlock(out_channels, reduction=se_reduction) if use_se else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se(self.block(x))


class UNet2D(nn.Module):
    """2D U-Net with four downsampling levels.

    Args:
        in_channels: Input channels. ``3`` for the multimodal setup
            (FLAIR, T1, T2) and for the 2.5D setup (slices z-1, z, z+1).
        out_channels: Output channels. ``1`` for binary segmentation.
        base_channels: Channel width of the first encoder level. The network
            doubles this at every level, up to ``base_channels * 16`` in the
            bottleneck.
        use_se: Whether to insert SE blocks. Set to ``False`` for the ablation.
        se_reduction: SE bottleneck ratio.

    Note:
        Input spatial dimensions must be divisible by 16 (four max-pools).
        The default preprocessing produces 224x192 slices, which satisfies this.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
        use_se: bool = True,
        se_reduction: int = 16,
    ) -> None:
        super().__init__()

        def conv(cin: int, cout: int) -> DoubleConv:
            return DoubleConv(cin, cout, use_se=use_se, se_reduction=se_reduction)

        c = base_channels

        # Encoder
        self.enc1 = conv(in_channels, c)
        self.enc2 = conv(c, c * 2)
        self.enc3 = conv(c * 2, c * 4)
        self.enc4 = conv(c * 4, c * 8)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = conv(c * 8, c * 16)

        # Decoder
        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, kernel_size=2, stride=2)
        self.dec4 = conv(c * 16, c * 8)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec3 = conv(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = conv(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = conv(c * 2, c)

        self.out_conv = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape ``(B, out_channels, H, W)``."""
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))

        xb = self.bottleneck(self.pool(x4))

        y = self.dec4(torch.cat([x4, self.up4(xb)], dim=1))
        y = self.dec3(torch.cat([x3, self.up3(y)], dim=1))
        y = self.dec2(torch.cat([x2, self.up2(y)], dim=1))
        y = self.dec1(torch.cat([x1, self.up1(y)], dim=1))

        return self.out_conv(y)

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
