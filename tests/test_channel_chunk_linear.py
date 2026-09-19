"""Rigid and affine registration with the loss evaluated in channel chunks, against all channels at once."""
from pathlib import Path

import pytest
import torch

from fireants.io.image import BatchedImages, FakeBatchedImages, Image
from fireants.registration.affine import AffineRegistration
from fireants.registration.rigid import RigidRegistration

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

TUTORIALS = Path(__file__).parent.parent / "tutorials"


def _images(masked):
    batches = []
    for name in (1000, 1001):
        image = Image.load_file(str(TUTORIALS / f"atlas_2mm_{name}_3.nii.gz"), device="cuda:0")
        x = image.array.float()
        x = (x - x.min()) / (x.max() - x.min())
        channels = [x, x.sqrt(), x * x] + ([(x > 0.05).float()] if masked else [])
        batches.append(FakeBatchedImages(torch.cat(channels, dim=1), BatchedImages([image])))
    return batches


def _matrix(cls, loss, chunk):
    fixed, moving = _images(loss.startswith("masked_"))
    reg = cls(scales=[4, 2, 1], iterations=[40, 30, 20], fixed_images=fixed, moving_images=moving,
              loss_type=loss, cc_kernel_size=5, optimizer="Adam", optimizer_lr=3e-3,
              progress_bar=False, channel_chunk=chunk)
    reg.optimize()
    matrix = reg.get_rigid_matrix() if cls is RigidRegistration else reg.get_affine_matrix()
    return matrix.detach().cpu()


@pytest.mark.parametrize("cls", [RigidRegistration, AffineRegistration])
@pytest.mark.parametrize("loss", ["cc", "masked_cc", "mse", "masked_mse"])
def test_chunked_matches_full(cls, loss):
    full, chunked = _matrix(cls, loss, None), _matrix(cls, loss, 1)
    assert (full - torch.eye(4)).abs().max() > 1e-3  # the registration does move things
    assert torch.allclose(full, chunked, atol=2e-3, rtol=0)


def test_rejects_global_loss():
    fixed, moving = _images(False)
    with pytest.raises(NotImplementedError):
        AffineRegistration(scales=[1], iterations=[1], fixed_images=fixed, moving_images=moving,
                           loss_type="mi", channel_chunk=1)
