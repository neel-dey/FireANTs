"""Rigid and affine registration with the loss evaluated in channel chunks, against all channels at once.

The chunks may come from GPU or host memory; both are measured against the same unchunked reference.
"""
from pathlib import Path

import pytest
import torch

from fireants.io.image import BatchedImages, FakeBatchedImages, Image
from fireants.registration.affine import AffineRegistration
from fireants.registration.rigid import RigidRegistration

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

TUTORIALS = Path(__file__).parent.parent / "tutorials"


def _images(masked, device="cuda:0", repeats=1):
    """Fixed and moving feature batches; `device` is where the channels live, not the geometry."""
    batches = []
    for name in (1000, 1001):
        image = Image.load_file(str(TUTORIALS / f"atlas_2mm_{name}_3.nii.gz"), device="cuda:0")
        x = image.array.float()
        x = (x - x.min()) / (x.max() - x.min())
        channels = [x, x.sqrt(), x * x] * repeats + ([(x > 0.05).float()] if masked else [])
        batches.append(FakeBatchedImages(torch.cat(channels, dim=1).to(device), BatchedImages([image])))
    return batches


def _matrix(cls, loss, chunk, device="cuda:0"):
    fixed, moving = _images(loss.startswith("masked_"), device)
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


@pytest.mark.parametrize("cls", [RigidRegistration, AffineRegistration])
@pytest.mark.parametrize("loss", ["cc", "masked_cc", "mse", "masked_mse"])
def test_host_resident_matches_full(cls, loss):
    """Channels kept in host memory, staged to the GPU a chunk at a time."""
    full, host = _matrix(cls, loss, None), _matrix(cls, loss, 1, device="cpu")
    assert (full - torch.eye(4)).abs().max() > 1e-3
    assert torch.allclose(full, host, atol=2e-3, rtol=0)


@pytest.mark.parametrize("cls", [RigidRegistration, AffineRegistration])
def test_host_arrays_never_reach_the_gpu_whole(cls):
    """Host-resident channels cost GPU memory a chunk at a time, and stay on the host."""
    def peak(device):
        fixed, moving = _images(False, device, repeats=8)  # 24 channels
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        reg = cls(scales=[2, 1], iterations=[4, 4], fixed_images=fixed, moving_images=moving,
                  loss_type="cc", cc_kernel_size=5, optimizer="Adam", optimizer_lr=3e-3,
                  progress_bar=False, channel_chunk=2)
        reg.optimize()
        assert reg.fixed_images().device.type == torch.device(device).type
        measured = torch.cuda.max_memory_allocated(), reg.fixed_images().numel() * 4
        del reg, fixed, moving
        return measured

    (on_gpu, volume), (on_host, _) = peak("cuda:0"), peak("cpu")
    # at the very least the two feature volumes the device path holds for the whole stage
    assert on_host < on_gpu - 2 * volume, f"host peak {on_host} vs device peak {on_gpu}"


def test_host_resident_needs_a_local_loss():
    fixed, moving = _images(False, device="cpu")
    reg = AffineRegistration(scales=[1], iterations=[1], fixed_images=fixed, moving_images=moving,
                             loss_type="mi", progress_bar=False)
    with pytest.raises(NotImplementedError):
        reg.optimize()


def test_rejects_global_loss():
    fixed, moving = _images(False)
    with pytest.raises(NotImplementedError):
        AffineRegistration(scales=[1], iterations=[1], fixed_images=fixed, moving_images=moving,
                           loss_type="mi", channel_chunk=1)
