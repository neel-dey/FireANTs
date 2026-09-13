"""Check block correspondences, robust affine fits, and CUDA agreement."""

import itertools

import numpy as np
import pytest
import torch

from fireants.registration.block_matching import fit_affine, fit_affine_lts, match_blocks, select_blocks


@pytest.fixture(autouse=True)
def limit_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def loop_matches(reference, warped, mask, origins):
    dims = reference.ndim
    shifts = list(itertools.product(range(-3, 4), repeat=dims))
    answers, scores = [], []
    for origin in origins:
        best, answer = -1., (0,) * dims
        for shift in shifts:
            a, b = [], []
            for offset in itertools.product(range(4), repeat=dims):
                point = tuple(int(origin[dims - 1 - d]) + offset[d] for d in range(dims))
                other = tuple(point[d] + shift[d] for d in range(dims))
                if any(p < 0 or p >= n for p, n in zip(point, reference.shape)):
                    continue
                if any(p < 0 or p >= n for p, n in zip(other, reference.shape)):
                    continue
                if mask[point] and np.isfinite(reference[point]) and np.isfinite(warped[other]):
                    a.append(reference[point]); b.append(warped[other])
            if len(a) <= 4**dims / 2:
                continue
            a, b = np.array(a, dtype=np.float64), np.array(b, dtype=np.float64)
            a -= a.mean(); b -= b.mean()
            va, vb = a @ a, b @ b
            if va <= 1e-12 or vb <= 1e-12:
                continue
            score = np.round(abs(a @ b) / np.sqrt(va * vb), 10)
            if score > best:
                best, answer = score, shift[::-1]
        answers.append(answer); scores.append(best)
    return np.array(answers), np.array(scores)


@pytest.mark.parametrize("dims", [2, 3])
def test_matches_against_independent_masked_search(dims):
    rng = np.random.default_rng(8)
    reference = rng.normal(size=(9,) * dims).astype(np.float32)
    moving = np.roll(reference, 1, axis=-1) * -2. + 5.
    mask = rng.random(reference.shape) > .15
    moving[(slice(None),) * (dims-1) + (0,)] = np.nan
    origins = np.array([[0]*dims, [4]*dims, [7]*dims])
    expected, scores = loop_matches(reference, moving, mask, origins)
    actual, values = match_blocks(torch.tensor(reference), torch.tensor(moving), torch.tensor(mask), torch.tensor(origins))
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(values, scores, atol=1e-10, rtol=0)


@pytest.mark.parametrize("dims", [2, 3])
def test_affine_large_physical_origins_and_outliers(dims):
    rng = np.random.default_rng(12)
    source = rng.normal(0, 30, (150, dims)) + 10000
    linear = np.eye(dims) + rng.normal(0, .08, (dims, dims))
    translation = rng.normal(0, 20, dims)
    target = source @ linear.T + translation
    matrix = fit_affine(torch.tensor(source), torch.tensor(target))
    np.testing.assert_allclose(matrix[:dims, :dims], linear, atol=1e-10)
    np.testing.assert_allclose(matrix[:dims, -1], translation, atol=1e-7)
    corrupted = target.copy()
    corrupted[:35] += rng.normal(0, 100, (35, dims))
    fitted = fit_affine_lts(torch.tensor(source), torch.tensor(corrupted))
    np.testing.assert_allclose(source @ fitted[:dims, :dims].numpy().T + fitted[:dims, -1].numpy(), target, atol=1e-6)


@pytest.mark.parametrize("dims", [2, 3])
def test_empty_masks_and_rank_deficiency(dims):
    image = torch.ones((12,) * dims)
    assert len(select_blocks(image, torch.zeros_like(image, dtype=torch.bool))) == 0
    origins = torch.zeros((1, dims), dtype=torch.long)
    shift, score = match_blocks(image, image, torch.ones_like(image, dtype=torch.bool), origins)
    assert score.item() == -1
    assert not shift.any()
    with pytest.raises(ValueError, match="Rank-deficient"):
        fit_affine(torch.ones((10, dims)), torch.ones((10, dims)))


@pytest.mark.parametrize("dims", [2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_matches_torch_and_solves_on_device(dims, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(17)
    image = torch.randn((17,) * dims, device="cuda", dtype=dtype)
    mask = torch.rand_like(image) > .1
    moving = torch.roll(image, 1, -1) * -3. + .7
    moving[..., 0] = float("nan")
    origins = select_blocks(image, mask, backend="torch")
    torch.testing.assert_close(select_blocks(image, mask, backend="cuda"), origins, rtol=0, atol=0)
    expected, values = match_blocks(image, moving, mask, origins, backend="torch")
    actual, scores = match_blocks(image, moving, mask, origins, backend="cuda")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(scores, values, rtol=0, atol=1e-10)
    source = torch.randn(200, dims, device="cuda", dtype=dtype) * 20 + 1000
    target = source @ (torch.eye(dims, device="cuda", dtype=dtype) + .1).T + 7
    target[:35] += 80 * torch.randn_like(target[:35])
    expected = fit_affine_lts(source, target, backend="torch")
    actual = fit_affine_lts(source, target, backend="cuda")
    assert actual.is_cuda
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    with pytest.raises(ValueError, match="Rank-deficient"):
        fit_affine(torch.ones((10, dims), device="cuda"), torch.ones((10, dims), device="cuda"), backend="cuda")


@pytest.mark.parametrize("dims", [2, 3])
def test_torch_registration_recovers_known_affine(dims):
    from experiments.polar_affine import synthetic_case, score
    from fireants.registration.block_matching import BlockMatchingRegistration
    case = synthetic_case(dims, 0, "cpu")
    initial = np.asarray(case["truth"]).copy()
    initial[:dims, -1] += 2.
    reg = BlockMatchingRegistration([2, 1], [4, 4], case["fixed_batch"], case["moving_batch"],
                                    init_affine=torch.tensor(initial, dtype=torch.float32)[None],
                                    block_fraction=.1, progress_bar=False)
    before = score(case, initial)["tre_mean_mm"]
    reg.optimize()
    after = score(case, reg.get_affine_matrix()[0].numpy())["tre_mean_mm"]
    assert after < 1. and after < before / 2
    assert all(row["reason"] == "completed" for row in reg.history)


@pytest.mark.parametrize("dims", [2, 3])
def test_cuda_registration_agrees_with_torch(dims):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from experiments.polar_affine import synthetic_case
    from fireants.registration.block_matching import BlockMatchingRegistration
    case = synthetic_case(dims, 2, "cuda")
    initial = torch.tensor(case["truth"], dtype=torch.float32, device="cuda")[None]
    initial[:, :dims, -1] += 2.
    before = initial.clone()
    matrices = []
    for backend in ("torch", "cuda"):
        reg = BlockMatchingRegistration([2, 1], [3, 3], case["fixed_batch"], case["moving_batch"],
                                        init_affine=initial, backend=backend, progress_bar=False)
        reg.optimize()
        assert all(row["reason"] == "completed" for row in reg.history)
        matrices.append(reg.get_affine_matrix()[0])
    torch.testing.assert_close(initial, before, atol=0, rtol=0)
    points = torch.tensor(case["points"], device="cuda", dtype=torch.float32)
    difference = matrices[1] - matrices[0]
    errors = (points @ difference[:dims, :dims].T + difference[:dims, -1]).norm(dim=-1)
    assert errors.max() < .05


def test_cuda_lsq_respects_current_stream():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        source = torch.randn(1000, 3, device="cuda", dtype=torch.float64)
        target = source * 1.2 + 3.
        matrix = fit_affine(source, target, backend="cuda")
    stream.synchronize()
    torch.testing.assert_close(matrix[:3, :3], torch.eye(3, device="cuda", dtype=torch.float64) * 1.2)
    torch.testing.assert_close(matrix[:3, -1], torch.full((3,), 3., device="cuda", dtype=torch.float64))
