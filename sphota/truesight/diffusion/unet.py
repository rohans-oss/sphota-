"""Epsilon-prediction U-Net for the conditional decoder (Step 10): a diffusers UNet2DModel.

Input  = cat([x_t, condition], dim=1): 3 channels of noisy image + 3 channels of coarse codec decode = 6 channels.
Output = the predicted noise eps (3 channels). Attention sits at the middle resolutions (the third down block, its
mirror in the up path, and the mid block), so it is cheap enough for 128 x 128 training crops. The network is fully
convolutional: after training on 128 x 128 crops it decodes whole images (H, W multiples of 8; we pad to 64).
"""
from diffusers import UNet2DModel

DOWN = ("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D")
UP = ("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D")
CHANNELS = (64, 128, 256, 256)          # the guide's width; (16, 32, 32, 32) is enough for CPU smoke tests
MIN_MULTIPLE = 8                        # 3 down-samplings: H and W must be divisible by 8


def make_unet(size=128, channels=CHANNELS, layers_per_block=2, in_channels=6, out_channels=3):
    """The conditional eps-network. ``size`` only fixes the nominal sample size (any multiple of 8 works at run time)."""
    if len(channels) != len(DOWN):
        raise ValueError(f"channels must have {len(DOWN)} entries, got {len(channels)}")
    return UNet2DModel(sample_size=size, in_channels=in_channels, out_channels=out_channels,
                       layers_per_block=layers_per_block, block_out_channels=tuple(channels),
                       down_block_types=DOWN, up_block_types=UP, norm_num_groups=norm_groups(channels))


def norm_groups(channels):
    """GroupNorm groups: 32 (diffusers default) when every width allows it, else the largest power of two that divides
    all widths (small CPU test networks)."""
    for g in (32, 16, 8, 4, 2, 1):
        if all(c % g == 0 for c in channels):
            return g


def unet_config(size=128, channels=CHANNELS, layers_per_block=2):
    """The arguments needed to rebuild the network from a checkpoint."""
    return dict(size=size, channels=tuple(channels), layers_per_block=layers_per_block)


def n_params(net):
    return sum(p.numel() for p in net.parameters())
