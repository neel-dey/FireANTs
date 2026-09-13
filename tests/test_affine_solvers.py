"""Check the affine experiment's coordinate conversion and L-BFGS recovery."""

import numpy as np
import pytest
import torch

from experiments.affine_solvers import fit_lbfgs, lps_ras, nifti_matrix
from experiments.polar_affine import score, synthetic_case
from fireants.registration.affine import AffineRegistration


@pytest.mark.parametrize("dims", [2, 3])
def test_nifti_matrix_maps_the_same_physical_points(dims):
    rng = np.random.default_rng(2)
    matrix = np.eye(dims + 1)
    matrix[:dims] += rng.normal(0, .1, (dims, dims + 1))
    point = rng.normal(0, 20, dims)
    expected = matrix[:dims, :dims] @ point + matrix[:dims, -1]
    expected[:2] *= -1
    ras_point = np.zeros(4)
    ras_point[:dims] = point
    ras_point[:2] *= -1
    ras_point[-1] = 1
    actual = nifti_matrix(matrix) @ ras_point
    np.testing.assert_allclose(actual[:dims], expected)
    np.testing.assert_allclose(lps_ras(lps_ras(matrix)), matrix)


@pytest.mark.parametrize("dims", [2, 3])
def test_lbfgs_recovers_known_image_transform(dims):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        case = synthetic_case(dims, 0, "cpu")
        initial = np.asarray(case["truth"]).copy()
        initial[:dims, -1] += 1.
        reg = AffineRegistration([2, 1], [50, 50], case["fixed_batch"], case["moving_batch"],
                                 loss_type="mse", init_rigid=torch.tensor(initial, dtype=torch.float32)[None],
                                 normalize_translation=True, progress_bar=False)
        trace = fit_lbfgs(reg, iterations=50)
        actual = reg.get_affine_matrix().detach()[0].numpy()
        assert score(case, actual)["tre_mean_mm"] < .1
        assert all(level["final_loss"] <= level["initial_loss"] + 1e-7 for level in trace)
        assert all(level["evaluations"] >= level["iterations"] for level in trace)
    finally:
        torch.set_num_threads(previous_threads)
