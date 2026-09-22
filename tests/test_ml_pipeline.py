import unittest

import numpy as np
import pandas as pd

from src.ml_pipeline import FEATURE_NAMES, prepare_features


class PrepareFeaturesTests(unittest.TestCase):
    def test_invalid_measurement_zeros_become_missing(self):
        row = {name: 1.0 for name in FEATURE_NAMES}
        row.update({"Glucose": 0, "Insulin": 0, "Outcome": 0})

        features, target = prepare_features(pd.DataFrame([row]))

        self.assertTrue(pd.isna(features.loc[0, "Glucose"]))
        self.assertTrue(pd.isna(features.loc[0, "Insulin"]))
        self.assertEqual(target.tolist(), [0])

    def test_non_binary_outcome_is_rejected(self):
        row = {name: 1.0 for name in FEATURE_NAMES}
        row["Outcome"] = 2

        with self.assertRaises(ValueError):
            prepare_features(pd.DataFrame([row]))

    def test_fractional_outcome_is_rejected(self):
        row = {name: 1.0 for name in FEATURE_NAMES}
        row["Outcome"] = 0.5

        with self.assertRaises(ValueError):
            prepare_features(pd.DataFrame([row]))

    def test_non_numeric_feature_is_rejected(self):
        row: dict[str, object] = {name: 1.0 for name in FEATURE_NAMES}
        row["Glucose"] = "unknown"
        row["Outcome"] = 0

        with self.assertRaises(ValueError):
            prepare_features(pd.DataFrame([row]))

    def test_out_of_range_feature_is_rejected(self):
        row = {name: 1.0 for name in FEATURE_NAMES}
        row.update({"Glucose": 301, "Outcome": 0})

        with self.assertRaises(ValueError):
            prepare_features(pd.DataFrame([row]))


if __name__ == "__main__":
    unittest.main()
