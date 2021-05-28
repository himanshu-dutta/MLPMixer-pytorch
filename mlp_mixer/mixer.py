import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, dim: int, expansion: int, dropout: float) -> None:
        super(MLP, self).__init__()
        self.lin1 = nn.Linear(dim, dim * expansion)
        self.act = nn.GELU()
        self.lin2 = nn.Linear(dim * expansion, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.lin2(x)
        x = self.drop(x)

        return x


class MixerLayer(nn.Module):
    def __init__(
        self, num_patches: int, num_channels: int, expansion: int, dropout: float
    ) -> None:
        super(MixerLayer, self).__init__()
        self.norm1 = nn.LayerNorm(num_channels)
        self.by_patch = MLP(num_patches, expansion, dropout)
        self.by_channel = MLP(num_channels, expansion, dropout)
        self.norm2 = nn.LayerNorm(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape -> B, P, C
        identity = x
        x = self.norm1(x)

        # x.shape -> B, C, P
        x = self.by_patch(torch.transpose(x, 1, 2))

        # x.shape -> B, P, C
        x = torch.transpose(x, 1, 2) + identity

        # x.shape -> B, P, C
        identity = x
        x = self.norm2(x)
        x = self.by_channel(x) + identity

        return x


class MLPMixer(nn.Module):
    r"""Pytorch implementation of [MLPMixer](https://arxiv.org/abs/2105.01601)::

        from mlp_mixer import MLPMixer

        model = MLPMixer(
                img_size=IMG_SZ,
                img_channels=IMG_CHANNELS,
                num_classes=NUM_CLASSES,
                mixer_depth=DEPTH,
                num_patches=NUM_PATCHES,
                num_channels=NUM_CHANNELS,
                expansion=EXPANSION,
                dropout=DROPOUT,
            )

    Parameters
    ----------
    `img_size` : `int`
        the input size of the image
    `img_channels` : `int`
        the number of channels in the input image
    `num_classes` : `int`
        the number of classes in the data
    `mixer_depth` : `int`
        the number of Mixer Layers in the model
    `num_patches` : `int`
        the number of patches per image
    `num_channels` : `int`
        number of channels for the `per channel fully connected`
    `expansion` : `int`
        expansion dim for fc layers
    `dropout` : `float`
    """

    def __init__(
        self,
        img_size: int,
        img_channels: int,
        num_classes: int,
        mixer_depth: int,
        num_patches: int,
        num_channels: int,
        expansion: int,
        dropout: float,
    ) -> None:
        super(MLPMixer, self).__init__()

        self.img_size = img_size
        self.img_channels = img_channels

        self.num_patches = num_patches
        self.num_channels = num_channels

        self.patch_sz = int(((self.img_size ** 2) // self.num_patches) ** (1 / 2))

        inp_channels = ((img_size ** 2) // num_patches) * img_channels

        self.per_patch = nn.Linear(inp_channels, num_channels)

        self.mixer_layers = nn.ModuleList(
            [
                MixerLayer(num_patches, num_channels, expansion, dropout)
                for _ in range(mixer_depth)
            ]
        )

        self.identity = nn.Identity()
        self.classifier = nn.Linear(num_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        bs = x.shape[0]

        # creating patches of the images
        # input shape -> BS, IMG_CHANNELS, IMG_SIZE, IMG_SIZE
        # output shape -> BS, NUM_PATCHES, NUM_CHANNELS
        x = (
            x.data.unfold(1, self.img_channels, self.img_channels)
            .unfold(2, self.patch_sz, self.patch_sz)
            .unfold(3, self.patch_sz, self.patch_sz)
        )
        x = x.reshape(bs, -1, self.img_channels * self.patch_sz * self.patch_sz)

        # per patch fc
        x = self.per_patch(x)

        # Mixer Layer x N
        for layer in self.mixer_layers:
            x = layer(x)

        # classifier
        x = x.mean(1)
        x = self.identity(x)  # for feature extraction
        return self.classifier(x)