"""ShardedGreedyRegistration against GreedyRegistration on the tutorial atlases.

Both start from the same small off-grid warp: at the identity every sample
falls exactly on a voxel, where the side of the interpolation gradient is
decided by coordinate rounding, which differs between a slab and the full grid.
"""
from pathlib import Path

import pytest
import torch

from fireants.io.image import BatchedImages, FakeBatchedImages, Image
from fireants.registration.greedy import GreedyRegistration
from fireants.registration.shardedgreedy import ShardedGreedyRegistration

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

TUTORIALS = Path(__file__).parent.parent / "tutorials"
SCALES, ITERATIONS = [4, 2, 1], [30, 20, 10]


def _load(device):
    images = [Image.load_file(str(TUTORIALS / f"atlas_2mm_{name}_3.nii.gz"), device=device) for name in (1000, 1001)]
    batches = []
    for image in images:
        x = image.array.float()
        x = (x - x.min()) / (x.max() - x.min())
        channels = torch.cat([x, x.sqrt(), x * x, (x > 0.05).float()], dim=1)  # last channel: mask
        batches.append((BatchedImages([image]), channels))
    return batches


def _register(cls, loss, device, **extra):
    (fixed_geom, fixed), (moving_geom, moving) = _load(device)
    if not loss.startswith("masked_"):
        fixed, moving = fixed[:, :-1].contiguous(), moving[:, :-1].contiguous()
    reg = cls(scales=SCALES, iterations=ITERATIONS,
              fixed_images=FakeBatchedImages(fixed, fixed_geom), moving_images=FakeBatchedImages(moving, moving_geom),
              loss_type=loss, cc_kernel_size=[7, 5, 5], optimizer="Adam", optimizer_lr=0.5,
              progress_bar=False, **extra)
    first = [max(int(s / SCALES[0]), 32) for s in fixed.shape[2:]]
    torch.manual_seed(0)
    start = 0.02 * torch.nn.functional.avg_pool3d(torch.randn(1, 3, *first), 5, 1, 2).permute(0, 2, 3, 4, 1)
    if cls is GreedyRegistration:
        reg.warp.warp.data.copy_(start)
    else:
        for shard in reg._shards:
            shard.warp.data.copy_(start.narrow(reg._vdim, shard.lo, shard.hi - shard.lo))
    reg.optimize()
    warp = reg.warp.get_warp() if cls is GreedyRegistration else reg.get_warp()
    voxels = (torch.tensor(warp.shape[1:-1][::-1], dtype=torch.float32) - 1) / 2
    return warp.detach().cpu() * voxels


def _mean_difference(a, b):
    return (a - b).norm(dim=-1).mean().item()


def _mi_setup(devices, loss, channel_chunk=None, scale=4):
    """A sharded registration at one level whose `_sample` is the identity.

    Every slab's moving image is replaced by a leaf tensor already on the fixed grid,
    so a comparison against the stock loss sees the histogram reduction alone and not
    the resampling. Returns the whole fixed and moved volumes and the leaves.
    """
    (fixed_geom, fixed), (moving_geom, moving) = _load("cpu")
    if not loss.startswith("masked_"):
        fixed, moving = fixed[:, :-1].contiguous(), moving[:, :-1].contiguous()
    reg = ShardedGreedyRegistration(scales=[scale], iterations=[1], devices=devices, loss_type=loss,
                                    fixed_images=FakeBatchedImages(fixed, fixed_geom),
                                    moving_images=FakeBatchedImages(moving, moving_geom),
                                    channel_chunk=channel_chunk, progress_bar=False)
    reg._prepare()
    reg._begin_level(scale, 1)
    whole = torch.cat([torch.cat([s.fixed[k].cpu() for s in reg._shards], dim=reg._idim)
                       for k in range(len(reg._shards[0].fixed))], dim=1)
    torch.manual_seed(0)
    moved = torch.roll(whole, 2, dims=reg._idim) + 0.05 * torch.rand_like(whole)  # correlated: MI is not near zero
    mask = (torch.rand(reg.opt_size, 1, *reg._level_size) > 0.3).float() if reg.masked else None
    leaves = []
    for shard in reg._shards:
        width = shard.hi - shard.lo
        shard.moving = [moved[:, c0:c1].narrow(reg._idim, shard.lo, width).to(shard.device).requires_grad_(True)
                        for c0, c1 in reg._channel_chunks(moved.shape[1])]
        leaves.append(shard.moving)
        if reg.masked:
            shard.moving_mask = mask.narrow(reg._idim, shard.lo, width).to(shard.device).contiguous()
    reg._sample = lambda shard, image: image
    return reg, whole, moved, mask, leaves


@pytest.mark.parametrize("loss", ["mi", "masked_mi"])
@pytest.mark.parametrize("devices,chunk", [(["cuda:0"], None), (["cuda:0"], 1), (["cuda:0", "cuda:1"], 2)])
def test_mi_matches_single_device(loss, devices, chunk):
    """The sharded histograms give the global mutual information, not a per-slab average."""
    if torch.cuda.device_count() < len(devices):
        pytest.skip("needs two GPUs")
    reg, whole, moved, mask, leaves = _mi_setup(devices, loss, chunk)
    sharded = reg._mi_loss_and_gradients()

    device = reg.devices[0]
    pred, target = moved.to(device).requires_grad_(True), whole.to(device)
    if reg.masked:
        fixed_mask = torch.cat([s.fixed_mask.cpu() for s in reg._shards], dim=reg._idim).to(device)
        reference = reg.loss_fn(torch.cat([pred, mask.to(device)], 1), torch.cat([target, fixed_mask], 1))
    else:
        reference = reg.loss_fn(pred, target)
    reference.backward()

    assert abs(sharded - reference.item()) <= 1e-5 * abs(reference.item())
    grad = torch.zeros_like(moved)
    for shard, parts in zip(reg._shards, leaves):
        for (c0, c1), leaf in zip(reg._channel_chunks(moved.shape[1]), parts):
            grad[:, c0:c1].narrow(reg._idim, shard.lo, shard.hi - shard.lo).copy_(leaf.grad.cpu())
    assert (grad - pred.grad.cpu()).norm().item() <= 1e-5 * pred.grad.norm().item()


@pytest.mark.parametrize("loss", ["cc", "masked_cc", "mse", "mi", "masked_mi"])
def test_one_device_matches_greedy(loss):
    reference = _register(GreedyRegistration, loss, "cuda:0")
    sharded = _register(ShardedGreedyRegistration, loss, "cpu", devices=["cuda:0"])
    chunked = _register(ShardedGreedyRegistration, loss, "cpu", devices=["cuda:0"], channel_chunk=1)
    assert _mean_difference(reference, sharded) < 1e-2
    assert _mean_difference(reference, chunked) < 5e-2


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
@pytest.mark.parametrize("loss", ["cc", "masked_cc", "mi", "masked_mi"])
@pytest.mark.parametrize("dim_to_shard", [0, 2])
def test_two_devices_match_greedy(loss, dim_to_shard):
    reference = _register(GreedyRegistration, loss, "cuda:0")
    sharded = _register(ShardedGreedyRegistration, loss, "cpu", devices=["cuda:0", "cuda:1"],
                        dim_to_shard=dim_to_shard, channel_chunk=2)
    assert reference.norm(dim=-1).mean().item() > 0.5  # the registration does move things
    assert _mean_difference(reference, sharded) < 5e-2


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
@pytest.mark.parametrize("loss", ["masked_mi", "masked_cc"])
def test_roi_confined_to_one_slab(loss):
    """A mask that misses a slab entirely: no voxel of it reaches the loss.

    Masked MI selects voxels, so such a slab gets no gradient at all; it still has
    to take the update, which its neighbour's smoothing halo reaches into.
    """
    def register(devices):
        (fixed_geom, fixed), (moving_geom, moving) = _load("cpu")
        for channels in (fixed, moving):
            channels[:, -1:] = 0
            channels[:, -1:, :channels.shape[2] // 3] = 1  # inside the first slab of axis 0
        reg = ShardedGreedyRegistration(
            scales=SCALES, iterations=ITERATIONS, devices=devices, dim_to_shard=0,
            fixed_images=FakeBatchedImages(fixed, fixed_geom),
            moving_images=FakeBatchedImages(moving, moving_geom),
            loss_type=loss, cc_kernel_size=[7, 5, 5], optimizer="Adam", optimizer_lr=0.5,
            progress_bar=False)
        reg.optimize()
        return reg.get_warp().detach().cpu()

    one, two = register(["cuda:0"]), register(["cuda:0", "cuda:1"])
    assert torch.isfinite(two).all()
    assert _mean_difference(one, two) < 5e-2


def test_rejects_unsplittable_loss():
    (fixed_geom, fixed), (moving_geom, moving) = _load("cpu")
    with pytest.raises(NotImplementedError):
        ShardedGreedyRegistration(scales=[1], iterations=[1], devices=["cuda:0"], loss_type="custom",
                                  custom_loss=torch.nn.MSELoss(),
                                  fixed_images=FakeBatchedImages(fixed[:, :1].contiguous(), fixed_geom),
                                  moving_images=FakeBatchedImages(moving[:, :1].contiguous(), moving_geom))
