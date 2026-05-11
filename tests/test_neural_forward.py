import torch

from hydroda.models.resunet import SmallResUNet
from hydroda.models.hyperda import HyperDA


def test_small_resunet_forward_shape():
    model = SmallResUNet(in_channels=12, out_channels=2, width=8)
    x = torch.randn(2, 12, 32, 48)
    y = model(x)
    assert y.shape == (2, 2, 32, 48)


def test_hyperda_generates_lightweight_blocks():
    block_shapes = {
        "adapter1.down_w": (4, 16, 1, 1),
        "adapter1.up_w": (16, 4, 1, 1),
        "head.delta_w": (2, 16, 1, 1),
    }
    model = HyperDA(prompt_feature_dim=6, block_shapes=block_shapes, hidden_dim=32, n_basis=3)
    prompt = torch.randn(2, 5, 6)
    zeta = model(prompt)
    assert set(zeta.keys()) == set(block_shapes.keys())
    assert zeta["head.delta_w"].shape == (2, 2, 16, 1, 1)
