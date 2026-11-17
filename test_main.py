"""Focused regression tests for the mission requirements and evaluation criteria."""

import contextlib
import io
import unittest
from unittest.mock import patch

import main


class TestLabelsAndDecision(unittest.TestCase):
    def test_label_aliases(self):
        self.assertEqual(main.normalize_label(" + "), "Cross")
        self.assertEqual(main.normalize_label("CROSS"), "Cross")
        self.assertEqual(main.normalize_label("plus"), "Cross")
        self.assertEqual(main.normalize_label("x"), "X")
        self.assertIsNone(main.normalize_label("circle"))

    def test_shared_epsilon_decision(self):
        self.assertEqual(main.decide(1.0, 1.0 + 1e-10), "UNDECIDED")
        self.assertEqual(main.decide(2.0, 1.0), "Cross")
        self.assertEqual(main.decide(1.0, 2.0), "X")

    def test_mode_one_displays_korean_tie_label(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            with patch("builtins.input", side_effect=["2", "1 1 1", "1 1 1", "1 1 1"]):
                main.user_mode()
        self.assertIn("판정 불가", output.getvalue())


class TestMacAndMatrices(unittest.TestCase):
    def test_mac_3x3_example(self):
        cross = main.make_cross(3)
        x_pattern = main.make_x(3)
        self.assertEqual(main.mac(cross, cross), 5.0)
        self.assertEqual(main.mac(x_pattern, cross), 1.0)
        self.assertIsInstance(main.mac(x_pattern, cross), float)

    def test_rejects_shape_and_invalid_values(self):
        with self.assertRaises(ValueError):
            main.mac([[1.0, 2.0]], [[1.0, 2.0]])
        with self.assertRaises(ValueError):
            main.mac([[float("nan")]], [[1.0]])
        with self.assertRaises(ValueError):
            main.mac([[True]], [[1.0]])

    def test_flat_mac_matches_two_dimensional(self):
        for size in (3, 5, 13, 25):
            pattern, filt = main.make_x(size), main.make_cross(size)
            self.assertAlmostEqual(main.mac(pattern, filt),
                                   main.mac_flat(main.flatten(pattern), main.flatten(filt)))


class TestSchemaAndGenerators(unittest.TestCase):
    def test_parse_size(self):
        self.assertEqual(main.parse_size("size_13"), 13)
        self.assertEqual(main.parse_size("size_13_2", pattern_key=True), 13)
        for key in ("size_0", "size_nope"):
            with self.assertRaises(ValueError):
                main.parse_size(key)

    def test_extract_filters_detects_duplicate_alias(self):
        with self.assertRaises(ValueError):
            main.extract_filter_pair({"cross": [], "+": [], "x": []})

    def test_check_case_and_schema_mismatch(self):
        filters = {3: {"Cross": main.make_cross(3), "X": main.make_x(3)}}
        case = {"input": main.make_x(3), "expected": "x"}
        self.assertTrue(main.check_case("size_3_1", case, filters)["ok"])
        with self.assertRaises(ValueError):
            main.check_case("size_3_2", {"input": [[1]], "expected": "x"}, filters)

    def test_patterns_have_expected_shape_and_self_score(self):
        for size in (3, 5, 13, 25):
            cross, x_pattern = main.make_cross(size), main.make_x(size)
            self.assertEqual(main.validate_matrix(cross), size)
            self.assertGreater(main.mac(cross, cross), main.mac(cross, x_pattern))
            self.assertGreater(main.mac(x_pattern, x_pattern), main.mac(x_pattern, cross))

    def test_table_counts_wide_characters(self):
        self.assertEqual(main.display_width("평균"), 4)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            main.print_table(("한글", "값"), (("Cross", "1.0"),))
        self.assertIn("Cross", output.getvalue())


if __name__ == "__main__":
    unittest.main()
