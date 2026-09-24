import torch

from member_a.standard_fsrcnn import StandardFSRCNN, macs_per_lr_pixel


def test_standard_fsrcnn_x2_shape_and_macs() -> None:
    model = StandardFSRCNN().eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 1, 27, 48))
    assert tuple(output.shape) == (1, 1, 54, 96)
    assert macs_per_lr_pixel() == 12_464
    assert sum(parameter.numel() for parameter in model.parameters()) == 12_809

