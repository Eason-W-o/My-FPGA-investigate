import torch

from member_a.original_fsrcnn_s import OriginalFSRCNNS, macs_per_lr_pixel


def test_original_fsrcnn_s_x2_shape_and_macs() -> None:
    model = OriginalFSRCNNS().eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 1, 27, 48))
    assert tuple(output.shape) == (1, 1, 54, 96)
    assert macs_per_lr_pixel() == 3937
