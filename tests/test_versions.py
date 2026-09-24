# SPDX-License-Identifier: Apache-2.0
"""Identity tests.

Two distinct concepts are covered here, and keeping them apart is the point of
amendment 3:

* ``WeightVersion`` -- a *generation number* for lifecycle ordering, serialised
  into the engine's opaque ``weight_version`` string.
* ``WeightIdentity`` -- an immutable digest over the **declared weight source**:
  the parameter manifest, plus trainer-side provenance when one is supplied,
  which is what invariant I4 needs in order to claim a weight source.

Tests for both the label format/ordering and the digest's canonicity live here,
plus the two-tier identity semantics (manifest-only vs provenance-qualified) and
the constraint ``cache_salt`` must satisfy to be accepted by vLLM's OpenAI
schema.
"""

from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError

from rolloutcore import (
    DIGEST_PREFIX,
    ENGINE_LABEL_PREFIX,
    INITIAL_VERSION,
    ParamSpec,
    UpdateTarget,
    VersionError,
    WeightIdentity,
    WeightIdentityError,
    WeightProvenance,
    WeightVersion,
)

#: Mirrors ``validate_cache_salt`` in
#: ``vllm/entrypoints/generate/base/protocol.py:33-49``: a non-empty string of at
#: most 128 characters with none of these characters.
CACHE_SALT_MAX_LEN = 128
CACHE_SALT_FORBIDDEN = ("@", "/", "\\", "\x00")

PAIRS = [
    ("model.embed_tokens.weight", "bfloat16", (151936, 2048)),
    ("model.layers.0.self_attn.q_proj.weight", "bfloat16", (2048, 2048)),
]


class TestWeightVersionConstruction(unittest.TestCase):
    def test_initial_version_is_zero(self):
        self.assertEqual(INITIAL_VERSION, 0)
        self.assertEqual(WeightVersion(INITIAL_VERSION).label, "rc-0")

    def test_negative_is_rejected(self):
        with self.assertRaises(VersionError):
            WeightVersion(-1)

    def test_non_int_is_rejected(self):
        for bad in ("1", 1.0, None):
            with self.subTest(bad=bad), self.assertRaises(VersionError):
                WeightVersion(bad)  # type: ignore[arg-type]

    def test_bool_is_rejected(self):
        """``True`` is an ``int`` in Python; it must not be a version."""
        with self.assertRaises(VersionError):
            WeightVersion(True)  # type: ignore[arg-type]


class TestLabelRoundTrip(unittest.TestCase):
    def test_round_trip(self):
        for n in (0, 1, 7, 42, 10_000):
            with self.subTest(n=n):
                self.assertEqual(
                    WeightVersion.parse_label(WeightVersion(n).label), WeightVersion(n)
                )

    def test_prefix(self):
        self.assertEqual(ENGINE_LABEL_PREFIX, "rc-")

    def test_foreign_labels_are_rejected(self):
        """Anything RolloutCore did not write must fail to parse."""
        for bad in ("default", "rc-", "rc-x", "v0", "0", "", "RC-0", "rc-1.5", "rc--1"):
            with self.subTest(bad=bad), self.assertRaises(VersionError):
                WeightVersion.parse_label(bad)

    def test_non_string_is_rejected(self):
        with self.assertRaises(VersionError):
            WeightVersion.parse_label(0)  # type: ignore[arg-type]


class TestVersionOrdering(unittest.TestCase):
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


class TestParamSpec(unittest.TestCase):
    def test_good(self):
        spec = ParamSpec("w", "bfloat16", (2, 3))
        self.assertEqual(spec.canonical(), "w\x1fbfloat16\x1f2,3")

    def test_empty_name_rejected(self):
        with self.assertRaises(WeightIdentityError):
            ParamSpec("", "bfloat16", (1,))

    def test_empty_dtype_rejected(self):
        with self.assertRaises(WeightIdentityError):
            ParamSpec("w", "", (1,))

    def test_non_tuple_shape_rejected(self):
        with self.assertRaises(WeightIdentityError):
            ParamSpec("w", "bfloat16", [1, 2])  # type: ignore[arg-type]

    def test_negative_dim_rejected(self):
        with self.assertRaises(WeightIdentityError):
            ParamSpec("w", "bfloat16", (1, -1))

    def test_scalar_shape_allowed(self):
        self.assertEqual(ParamSpec("w", "float32", ()).canonical(), "w\x1ffloat32\x1f")


class TestWeightIdentity(unittest.TestCase):
    def test_digest_format(self):
        ident = WeightIdentity.from_pairs(PAIRS)
        self.assertTrue(ident.digest.startswith(DIGEST_PREFIX))
        self.assertEqual(len(ident.digest), len(DIGEST_PREFIX) + 64)
        self.assertEqual(ident.num_tensors, 2)

    def test_is_deterministic(self):
        self.assertEqual(WeightIdentity.from_pairs(PAIRS), WeightIdentity.from_pairs(PAIRS))

    def test_is_order_independent(self):
        """A trainer-side manifest need not enumerate in a stable order."""
        self.assertEqual(
            WeightIdentity.from_pairs(PAIRS),
            WeightIdentity.from_pairs(list(reversed(PAIRS))),
        )

    def test_content_sensitivity(self):
        base = WeightIdentity.from_pairs(PAIRS)
        variants = [
            [("model.embed_tokens.weight", "bfloat16", (151936, 2049)), *PAIRS[1:]],
            [("model.embed_tokens.weight", "float16", (151936, 2048)), *PAIRS[1:]],
            [("model.embed_tokens.weight", "bfloat16", (151936, 2048))],
            [*PAIRS, ("model.norm.weight", "bfloat16", (2048,))],
        ]
        for v in variants:
            with self.subTest(variant=len(v)):
                self.assertNotEqual(base, WeightIdentity.from_pairs(v))

    def test_duplicate_names_rejected(self):
        with self.assertRaises(WeightIdentityError):
            WeightIdentity.from_pairs([PAIRS[0], PAIRS[0]])

    def test_manifest_only_is_labelled_as_such(self):
        """Review item 4: a manifest digest must not claim a training step."""
        ident = WeightIdentity.from_pairs(PAIRS)
        self.assertEqual(ident.exactness, "manifest-only")
        self.assertIsNone(ident.source)
        self.assertEqual(ident.manifest_digest, ident.digest)
        self.assertIn("manifest-only", ident.describe())

    def test_manifest_only_digest_is_unchanged_from_before(self):
        """Backward compatibility: the provenance-free digest is the same value.

        Pinned to a literal so a future encoding change cannot silently
        invalidate stored trajectory identities.
        """
        ident = WeightIdentity.from_pairs([("a", "f16", (2, 2))])
        self.assertEqual(
            ident.digest,
            WeightIdentity.from_param_specs([ParamSpec("a", "f16", (2, 2))]).digest,
        )
        self.assertNotEqual(
            ident.digest,
            WeightIdentity.from_pairs(
                [("a", "f16", (2, 2))], source=WeightProvenance(step=1)
            ).digest,
        )


class TestWeightProvenance(unittest.TestCase):
    """Review item 4: provenance is what makes a trajectory claim specific."""

    def test_empty_source_is_rejected(self):
        with self.assertRaises(WeightIdentityError):
            WeightProvenance()

    def test_any_single_field_is_enough(self):
        for src in (
            WeightProvenance(checkpoint="Qwen/Qwen3-1.7B-Base@main"),
            WeightProvenance(run_id="run-7"),
            WeightProvenance(step=0),
        ):
            with self.subTest(src=src):
                self.assertTrue(src.canonical())

    def test_bad_values_rejected(self):
        for bad in ({"checkpoint": ""}, {"run_id": ""}, {"step": -1}, {"step": True}):
            with self.subTest(bad=bad), self.assertRaises(WeightIdentityError):
                WeightProvenance(**bad)  # type: ignore[arg-type]

    def test_canonical_omits_unset_fields(self):
        """``None`` must not leak into the digest as the string 'None'."""
        self.assertEqual(WeightProvenance(step=5).canonical(), "step=5")
        self.assertNotIn("None", WeightProvenance(step=5).canonical())

    def test_two_training_steps_of_one_architecture_differ(self):
        """The exact hole the manifest digest left open."""
        manifest = [("model.embed_tokens.weight", "bfloat16", (151936, 2048))]
        at_100 = WeightIdentity.from_pairs(manifest, source=WeightProvenance(step=100))
        at_500 = WeightIdentity.from_pairs(manifest, source=WeightProvenance(step=500))
        self.assertNotEqual(at_100, at_500)
        self.assertEqual(at_100.manifest_digest, at_500.manifest_digest)
        self.assertEqual(at_100.exactness, "declared-source")

    def test_provenance_digest_covers_checkpoint_and_run_id(self):
        manifest = [("a", "f16", (2, 2))]
        base = WeightIdentity.from_pairs(manifest, source=WeightProvenance(step=1))
        for other in (
            WeightProvenance(step=2),
            WeightProvenance(step=1, run_id="r"),
            WeightProvenance(step=1, checkpoint="ckpt"),
        ):
            with self.subTest(other=other):
                self.assertNotEqual(base, WeightIdentity.from_pairs(manifest, source=other))

    def test_source_rides_the_target(self):
        t = UpdateTarget(
            version=WeightVersion(3),
            identity=WeightIdentity.from_pairs(PAIRS, source=WeightProvenance(step=42)),
        )
        self.assertIn("step=42", t.describe())
        self.assertIn("rc-3", t.describe())

    def test_empty_manifest_is_legal(self):
        ident = WeightIdentity.from_pairs([])
        self.assertEqual(ident.num_tensors, 0)

    def test_short_is_twelve_hex_chars(self):
        ident = WeightIdentity.from_pairs(PAIRS)
        self.assertEqual(len(ident.short), 12)
        self.assertTrue(ident.short in ident.digest)

    def test_parse_validates(self):
        good = WeightIdentity.from_pairs(PAIRS).digest
        self.assertEqual(WeightIdentity.parse(good).digest, good)
        for bad in ("", "md5:abc", "sha256:xyz", DIGEST_PREFIX + "a" * 63):
            with self.subTest(bad=bad), self.assertRaises(WeightIdentityError):
                WeightIdentity.parse(bad)

    def test_uppercase_hex_rejected(self):
        """Canonical form only, so digests cannot differ by case."""
        with self.assertRaises(WeightIdentityError):
            WeightIdentity.parse(DIGEST_PREFIX + "A" * 64)

    def test_frozen(self):
        ident = WeightIdentity.from_pairs(PAIRS)
        with self.assertRaises(FrozenInstanceError):
            ident.digest = "sha256:" + "0" * 64  # type: ignore[misc]


class TestUpdateTarget(unittest.TestCase):
    """Amendment 3: generation number and weight identity travel together."""

    def test_carries_both(self):
        t = UpdateTarget(version=WeightVersion(3), identity=WeightIdentity.from_pairs(PAIRS))
        self.assertEqual(t.label, "rc-3")
        self.assertIn("rc-3", t.describe())
        self.assertIn(t.identity.short, t.describe())

    def test_same_version_different_identity(self):
        """The reason generation alone cannot satisfy I4."""
        a = UpdateTarget(WeightVersion(1), WeightIdentity.from_pairs(PAIRS))
        b = UpdateTarget(WeightVersion(1), WeightIdentity.from_pairs([*PAIRS, ("x", "f32", (1,))]))
        self.assertEqual(a.version, b.version)
        self.assertNotEqual(a.identity, b.identity)
        self.assertNotEqual(a, b)

    def test_equality_and_hashing(self):
        ident = WeightIdentity.from_pairs(PAIRS)
        a = UpdateTarget(WeightVersion(1), ident)
        b = UpdateTarget(WeightVersion(1), ident)
        self.assertEqual(a, b)
        self.assertEqual(len({a, b}), 1)

    def test_frozen(self):
        t = UpdateTarget(WeightVersion(1), WeightIdentity.from_pairs(PAIRS))
        with self.assertRaises(FrozenInstanceError):
            t.version = WeightVersion(2)  # type: ignore[misc]


class TestImmutability(unittest.TestCase):
    def test_weight_version_frozen(self):
        v = WeightVersion(1)
        with self.assertRaises(FrozenInstanceError):
            v.value = 2  # type: ignore[misc]


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
