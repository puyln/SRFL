from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from srfl.core_numpy import (
    gua_aggregate,
    gua_weights,
    project_simplex,
    sample_spherical_gaussian,
    solve_unbalanced_transport,
)


class TestUSGR(unittest.TestCase):
    def test_spherical_samples_have_unit_norm(self) -> None:
        rng = np.random.default_rng(7)
        samples = sample_spherical_gaussian(64, 12, rng=rng)
        np.testing.assert_allclose(
            np.linalg.norm(samples, axis=1), np.ones(64), atol=1e-10
        )

    def test_transport_is_finite_nonnegative_and_dimensionally_correct(self) -> None:
        rng = np.random.default_rng(11)
        features = rng.normal(size=(20, 8))
        result = solve_unbalanced_transport(
            features,
            random_sample_count=13,
            iterations=150,
            rng=rng,
        )
        self.assertEqual(result.coupling.shape, (20, 13))
        self.assertEqual(result.cost_matrix.shape, (20, 13))
        self.assertTrue(np.isfinite(result.loss))
        self.assertGreaterEqual(result.loss, 0.0)
        self.assertTrue((result.coupling >= 0).all())

    def test_transport_is_deterministic_given_seed(self) -> None:
        features = np.arange(60, dtype=float).reshape(10, 6)
        first = solve_unbalanced_transport(
            features, random_sample_count=7, rng=np.random.default_rng(5)
        )
        second = solve_unbalanced_transport(
            features, random_sample_count=7, rng=np.random.default_rng(5)
        )
        np.testing.assert_allclose(first.coupling, second.coupling)
        self.assertAlmostEqual(first.loss, second.loss)


class TestGUA(unittest.TestCase):
    def test_simplex_projection(self) -> None:
        projected = project_simplex(np.array([-1.0, 0.2, 2.1, 0.3]))
        self.assertTrue((projected >= 0).all())
        self.assertAlmostEqual(float(projected.sum()), 1.0, places=12)

    def test_gua_weights_and_gram(self) -> None:
        rng = np.random.default_rng(17)
        global_vector = rng.normal(size=31)
        clients = rng.normal(size=(5, 31))
        weights, gram, gradients, objective = gua_weights(
            global_vector, clients
        )
        self.assertEqual(weights.shape, (5,))
        self.assertEqual(gram.shape, (5, 5))
        self.assertEqual(gradients.shape, (5, 31))
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=10)
        self.assertTrue((weights >= 0).all())
        np.testing.assert_allclose(gram, gram.T, atol=1e-12)
        self.assertGreaterEqual(objective, -1e-12)

    def test_aggregate_shapes(self) -> None:
        global_vector = np.zeros(9)
        clients = np.stack(
            [np.full(9, -1.0), np.full(9, 0.5), np.full(9, 2.0)]
        )
        result = gua_aggregate(global_vector, clients, global_step=0.02)
        self.assertEqual(result.parameters.shape, (9,))
        self.assertEqual(result.direction.shape, (9,))
        self.assertTrue(np.isfinite(result.parameters).all())


class TestNumericalEdgeCases(unittest.TestCase):
    def test_transport_objective_decreases_with_more_fixed_steps(self):
        rng = np.random.default_rng(12)
        features = rng.normal(size=(7, 4))
        samples = rng.normal(size=(5, 4))
        losses = [solve_unbalanced_transport(features, random_samples=samples,
                  iterations=n).loss for n in (1, 10, 100)]
        self.assertTrue(np.all(np.diff(losses) <= 1e-12))

    def test_identical_clients_leave_global_parameters_unchanged(self):
        vector = np.arange(5, dtype=float)
        result = gua_aggregate(vector, np.tile(vector, (3, 1)))
        np.testing.assert_array_equal(result.parameters, vector)
        np.testing.assert_allclose(result.weights, np.ones(3) / 3)

    def test_single_client_uses_analytic_direction(self):
        result = gua_aggregate(np.zeros(2), np.array([[3., 4.]]), global_step=0.1)
        np.testing.assert_allclose(result.parameters, [0.024, 0.032])
        np.testing.assert_allclose(result.weights, [1.])

    def test_empty_or_nonfinite_inputs_fail_explicitly(self):
        cases = [lambda: project_simplex(np.array([])),
                 lambda: project_simplex(np.array([np.nan])),
                 lambda: solve_unbalanced_transport(np.ones((2, 3)), random_samples=np.empty((0, 3))),
                 lambda: gua_weights(np.ones(2), np.empty((0, 2))),
                 lambda: gua_weights(np.array([np.nan, 0.]), np.ones((2, 2)))]
        for call in cases:
            with self.assertRaises(ValueError):
                call()


if __name__ == "__main__":
    unittest.main()
