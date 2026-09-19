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


@pytest.mark.parametrize("loss", ["cc", "masked_cc", "mse"])
def test_one_device_matches_greedy(loss):
    reference = _register(GreedyRegistration, loss, "cuda:0")
    sharded = _register(ShardedGreedyRegistration, loss, "cpu", devices=["cuda:0"])
    chunked = _register(ShardedGreedyRegistration, loss, "cpu", devices=["cuda:0"], channel_chunk=1)
    assert _mean_difference(reference, sharded) < 1e-2
    assert _mean_difference(reference, chunked) < 5e-2


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
@pytest.mark.parametrize("loss", ["cc", "masked_cc"])
@pytest.mark.parametrize("dim_to_shard", [0, 2])
def test_two_devices_match_greedy(loss, dim_to_shard):
    reference = _register(GreedyRegistration, loss, "cuda:0")
    sharded = _register(ShardedGreedyRegistration, loss, "cpu", devices=["cuda:0", "cuda:1"],
                        dim_to_shard=dim_to_shard, channel_chunk=2)
    assert reference.norm(dim=-1).mean().item() > 0.5  # the registration does move things
    assert _mean_difference(reference, sharded) < 5e-2


def test_rejects_global_loss():
    (fixed_geom, fixed), (moving_geom, moving) = _load("cpu")
    with pytest.raises(NotImplementedError):
        ShardedGreedyRegistration(scales=[1], iterations=[1], devices=["cuda:0"], loss_type="mi",
                                  fixed_images=FakeBatchedImages(fixed[:, :1].contiguous(), fixed_geom),
                                  moving_images=FakeBatchedImages(moving[:, :1].contiguous(), moving_geom))
