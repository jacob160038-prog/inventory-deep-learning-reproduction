from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from src.reproduction import parse_array, reproduce_field_experiment, reproduce_offline


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"


class TestCPUReproduction(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.offline, _ = reproduce_offline(DATA)
        cls.post, cls.regressions, cls.did, _ = reproduce_field_experiment(DATA)

    def test_array_parser(self) -> None:
        np.testing.assert_allclose(parse_array("[1.0, 2, -3.5]"), [1.0, 2.0, -3.5])

    def test_section4_matches_saved_author_output(self) -> None:
        expected = {
            "OPT": 2572.964327,
            "E2E_RNN": 3036.758343,
            "Bm1": 3842.305332,
            "Bm2": 3837.828539,
            "Normal": 3961.691600,
            "Gamma": 4099.949751,
            "gbm": 3424.869965,
        }
        actual = self.offline.set_index("method")["total_cost"].to_dict()
        for method, value in expected.items():
            self.assertAlmostEqual(actual[method], value, places=6)

    def test_field_post_means_match_saved_author_output(self) -> None:
        values = self.post.set_index("metric")
        self.assertAlmostEqual(values.loc["holding_cost", "treatment_mean"], 497.99, places=6)
        self.assertAlmostEqual(values.loc["stockout_cost", "treatment_mean"], 386.10, places=6)
        self.assertAlmostEqual(values.loc["total_cost", "control_mean"], 1985.148, places=6)
        self.assertAlmostEqual(values.loc["stockout_rate", "control_mean"], 0.23652916666666668, places=12)

    def test_did_direction(self) -> None:
        self.assertTrue((self.did["did"] < 0).all())

    def test_both_regression_specifications_are_reported(self) -> None:
        specs = set(self.regressions["specification"])
        self.assertEqual(specs, {"released_notebook_no_intercept", "paper_equation_11_with_intercept"})


if __name__ == "__main__":
    unittest.main()
