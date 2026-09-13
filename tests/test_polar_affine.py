"""Geometry and optimization checks for polar affine parameters."""

import numpy as np
import pytest
import SimpleITK as sitk
import torch
from scipy.linalg import expm

from fireants.io import Image, BatchedImages
from fireants.registration.affine import AffineRegistration
from fireants.registration.polar import PolarLinear


@pytest.fixture(autouse=True)
def small_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def images(dims, count=1):
    data = np.zeros((24,) * dims, dtype=np.float32)
    image = sitk.GetImageFromArray(data)
    image.SetSpacing([1.2 + i * .3 for i in range(dims)])
    image.SetOrigin([30. + i * 20. for i in range(dims)])
    return BatchedImages([Image(image, device="cpu") for _ in range(count)])


@pytest.mark.parametrize("dims", [2, 3])
def test_identity_gradients_and_independent_exponential(dims):
    linear = PolarLinear(torch.eye(dims, dtype=torch.float64)[None])
    parameters = dict(linear.named_parameters())
    assert torch.autograd.gradcheck(
        lambda *p: torch.func.functional_call(linear, dict(zip(parameters, p)), ()),
        tuple(parameters.values()),
    )
    with torch.no_grad():
        linear.log_stretch.copy_(torch.linspace(-.2, .3, linear.log_stretch.numel()))
    expected = torch.from_numpy(expm(linear.symmetric()[0].detach().numpy()))[None]
    torch.testing.assert_close(linear(), expected)
    torch.testing.assert_close(linear.symmetric().square().sum(), linear.log_stretch.square().sum())


@pytest.mark.parametrize("dims", [2, 3])
@pytest.mark.parametrize("centered", [False, True])
@pytest.mark.parametrize("normalized", [False, True])
def test_initial_affine_and_reflection_are_preserved(dims, centered, normalized):
    fixed, moving = images(dims), images(dims, 2)
    matrix = torch.eye(dims + 1)[None].repeat(2, 1, 1)
    matrix[:, 0, 0] = 1.2
    matrix[:, 1, 1] = .8
    matrix[:, 0, 1] = .3
    matrix[1, 0, :dims] *= -1
    matrix[:, :dims, -1] = torch.arange(dims) + 3.
    original = matrix.clone()
    reg = AffineRegistration([1], [0], fixed, moving, loss_type="mse",
                             init_rigid=matrix, around_center=centered,
                             normalize_translation=normalized, parameterization="polar")
    torch.testing.assert_close(matrix, original, atol=0, rtol=0)
    torch.testing.assert_close(reg.get_affine_matrix(), original, atol=3e-5, rtol=1e-5)
    assert len(reg.optimized_parameters()) == 3
    assert all(p.is_leaf for p in reg.optimized_parameters())


@pytest.mark.parametrize("dims", [2, 3])
def test_scale_projection_preserves_center_and_orientation(dims):
    fixed = images(dims)
    reg = AffineRegistration([1], [0], fixed, fixed, loss_type="mse",
                             parameterization="polar", normalize_translation=True,
                             scale_bounds=(.75, 1.4), init_rigid="cof")
    with torch.no_grad():
        reg.polar.log_stretch[0, :dims] = torch.linspace(-2., 2., dims)
        reg.polar.log_stretch[0, dims:] = .5
        reg.transl.fill_(.1)
    before = reg.get_affine_matrix().detach()
    reg.project_scale()
    after = reg.get_affine_matrix().detach()
    center = torch.cat([reg.center, torch.ones(1, 1)], dim=-1)[..., None]
    torch.testing.assert_close(before @ center, after @ center, atol=3e-5, rtol=1e-5)
    scales = torch.linalg.svdvals(after[:, :dims, :dims])
    assert scales.min() >= .75 - 1e-6
    assert scales.max() <= 1.4 + 1e-6
    assert torch.linalg.det(after).item() > 0


@pytest.mark.parametrize("dims", [2, 3])
def test_optimizer_fits_rotation_scale_shear_and_translation(dims):
    fixed = images(dims)
    reg = AffineRegistration([1], [0], fixed, fixed, loss_type="mse",
                             parameterization="polar", normalize_translation=True,
                             optimizer_lr=.02)
    generator = torch.Generator().manual_seed(12)
    points = torch.randn(1, 100, dims, generator=generator) * 15 + reg.center[:, None]
    target_linear = torch.eye(dims)
    target_linear[0, 0] = 1.15
    target_linear[1, 1] = .88
    target_linear[0, 1] = .2
    target_linear[1, 0] = -.1
    target = (points - reg.center[:, None]) @ target_linear.T + reg.center[:, None] + 2
    for _ in range(250):
        reg.optimizer.zero_grad()
        affine = reg.get_affine_matrix(False)
        actual = points @ affine[:, :, :dims].mT + affine[:, None, :, -1]
        loss = (actual - target).square().mean()
        loss.backward()
        reg.optimizer.step()
    assert loss.item() < 1e-6


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_exponential_precision(dtype):
    linear = PolarLinear(torch.eye(3, dtype=dtype)[None])
    matrix = linear()
    torch.testing.assert_close(matrix, torch.eye(3, dtype=dtype)[None])
    matrix.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in linear.parameters())


def test_invalid_initialization():
    for matrix in [torch.zeros(1, 3, 3), torch.full((1, 3, 3), float("nan"))]:
        with pytest.raises(ValueError, match="initialization"):
            PolarLinear(matrix)


@pytest.mark.parametrize("magnitude", [0., 1e-12, 1., 1e6])
def test_quaternion_normalization_preserves_rotation(magnitude):
    linear = PolarLinear(torch.eye(3)[None])
    with torch.no_grad():
        linear.rotation.copy_(magnitude * torch.tensor([[.5, .5, .5, .5]]))
    matrix = linear()
    torch.testing.assert_close(matrix.mT @ matrix, torch.eye(3)[None])
    torch.testing.assert_close(torch.linalg.det(matrix), torch.ones(1))


@pytest.mark.parametrize("dims", [2, 3])
@pytest.mark.parametrize("optimizer", ["Adam", "SGD"])
def test_image_optimization_is_independent_of_physical_units(dims, optimizer):
    coords = np.indices((32,) * dims)
    data = np.exp(-sum((coords[i] - (12 + 2*i))**2 for i in range(dims)) / 40).astype("float32")
    grids = []
    for unit in (1., 1000.):
        batches = []
        for array in (data, np.roll(data, 2, axis=-1)):
            image = sitk.GetImageFromArray(array)
            image.SetSpacing([unit] * dims)
            batches.append(BatchedImages([Image(image, device="cpu")]))
        fixed, moving = batches
        reg = AffineRegistration([2, 1], [8, 8], fixed, moving, loss_type="mse",
                                 parameterization="polar", normalize_translation=True,
                                 optimizer=optimizer, optimizer_lr=.01, keep_best=True,
                                 scale_bounds=(.75, 1.4), progress_bar=False)
        reg.optimize()
        grids.append(reg.get_warped_coordinates(fixed, moving).detach())
    torch.testing.assert_close(*grids, atol=3e-5, rtol=3e-5)


def test_best_iterate_restores_polar_parameters():
    fixed = images(2)
    reg = AffineRegistration([1], [0], fixed, fixed, loss_type="mse",
                             parameterization="polar", keep_best=True)
    before = reg.get_affine_matrix().detach().clone()
    best = reg.best_iterate(reg.optimized_parameters())
    best.measured(0.)
    with torch.no_grad():
        for parameter in reg.optimized_parameters():
            parameter.add_(.2)
    best.measured(1.)
    assert best.restore()
    torch.testing.assert_close(reg.get_affine_matrix(), before)


@pytest.mark.parametrize("dims", [2, 3])
def test_export_matches_physical_point_mapping(dims, tmp_path):
    fixed = images(dims)
    reg = AffineRegistration([1], [0], fixed, fixed, loss_type="mse",
                             parameterization="polar", normalize_translation=True)
    with torch.no_grad():
        reg.polar.log_stretch.fill_(.12)
        reg.polar.rotation.add_(.1)
        reg.transl.fill_(.07)
    filename = str(tmp_path / "affine.mat")
    reg.save_as_ants_transforms(filename)
    transform = sitk.ReadTransform(filename)
    point = np.arange(dims, dtype=float) * 12. + 40.
    actual = transform.TransformPoint(point.tolist())
    matrix = reg.get_affine_matrix().detach()[0].numpy()
    expected = matrix[:dims, :dims] @ point + matrix[:dims, -1]
    np.testing.assert_allclose(actual, expected, atol=1e-5)
