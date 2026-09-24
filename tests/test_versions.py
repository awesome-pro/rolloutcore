# SPDX-License-Identifier: Apache-2.0
"""Version identity tests.

The engine treats ``weight_version`` as an opaque string
(``vllm/v1/engine/core.py:136-137``), so all type safety has to live here. These
tests pin the label format, the ordering, and the constraint that ``cache_salt``
must satisfy to be accepted by vLLM's OpenAI schema.
"""

from __future__ import annotations

import unittest

from rolloutcore import ENGINE_LABEL_PREFIX, INITIAL_VERSION, VersionError, WeightVersion

#: Mirrors ``validate_cache_salt`` in
#: ``vllm/entrypoints/generate/base/protocol.py:33-49``: a non-empty string of at
#: most 128 characters with none of these characters.
CACHE_SALT_MAX_LEN = 128
CACHE_SALT_FORBIDDEN = ("@", "/", "\\", "\x00")


class TestConstruction(unittest.TestCase):
    def test_initial_version_is_zero(self):
        self.assertEqual(INITIAL_VERSION, 0)
        self.assertEqual(WeightVersion(INITIAL_VERSION).label, "rc-0")

    def test_negative_is_rejected(self):
        with self.assertRaises(VersionError):
            WeightVersion(-1)

    def test_non_int_is_rejected(self):
        for bad in ("1", 1.0, None):
            with self.subTest(bad=bad):
                with self.assertRaises(VersionError):
                    WeightVersion(bad)  # type: ignore[arg-type]

    def test_bool_is_rejected(self):
        """``True`` is an ``int`` in Python; it must not be a version."""
        with self.assertRaises(VersionError):
            WeightVersion(True)  # type: ignore[arg-type]


class TestLabelRoundTrip(unittest.TestCase):
    def test_round_trip(self):
        for n in (0, 1, 7, 42, 10_000):
            with self.subTest(n=n):
                self.assertEqual(WeightVersion.parse_label(WeightVersion(n).label), WeightVersion(n))

    def test_prefix(self):
        self.assertEqual(ENGINE_LABEL_PREFIX, "rc-")

    def test_foreign_labels_are_rejected(self):
        """Anything RolloutCore did not write must fail to parse."""
        for bad in ("default", "rc-", "rc-x", "v0", "0", "", "RC-0", "rc-1.5", "rc--1"):
            with self.subTest(bad=bad):
                with self.assertRaises(VersionError):
                    WeightVersion.parse_label(bad)

    def test_non_string_is_rejected(self):
        with self.assertRaises(VersionError):
            WeightVersion.parse_label(0)  # type: ignore[arg-type]


class TestOrdering(unittest.TestCase):
    def test_ordering(self):
        self.assertLess(WeightVersion(1), WeightVersion(2))
        self.assertGreater(WeightVersion(5), WeightVersion(0))
        self.assertEqual(WeightVersion(3), WeightVersion(3))

    def test_next(self):
        self.assertEqual(WeightVersion(0).next(), WeightVersion(1))
        self.assertEqual(WeightVersion(41).next(), WeightVersion(42))

    def test_sortable(self):
        vs = [WeightVersion(n) for n in (5, 0, 3, 1)]
        self.assertEqual(sorted(vs), [WeightVersion(n) for n in (0, 1, 3, 5)])

    def test_hashable_and_usable_as_key(self):
        d = {WeightVersion(1): "a", WeightVersion(2): "b"}
        self.assertEqual(d[WeightVersion(1)], "a")


class TestCacheSalt(unittest.TestCase):
    def test_salt_is_the_label(self):
        for n in (0, 3, 99):
            with self.subTest(n=n):
                self.assertEqual(WeightVersion(n).cache_salt, WeightVersion(n).label)

    def test_distinct_versions_have_distinct_salts(self):
        salts = [WeightVersion(n).cache_salt for n in range(200)]
        self.assertEqual(len(set(salts)), len(salts))

    def test_satisfies_vllm_schema_constraint(self):
        """A version label must always be a legal ``cache_salt`` value."""
        for n in (0, 1, 42, 10_000, 10**12):
            with self.subTest(n=n):
                salt = WeightVersion(n).cache_salt
                self.assertTrue(salt)
                self.assertLessEqual(len(salt), CACHE_SALT_MAX_LEN)
                for ch in CACHE_SALT_FORBIDDEN:
                    self.assertNotIn(ch, salt)


class TestImmutability(unittest.TestCase):
    def test_frozen(self):
        v = WeightVersion(1)
        with self.assertRaises(Exception):
            v.value = 2  # type: ignore[misc]


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
