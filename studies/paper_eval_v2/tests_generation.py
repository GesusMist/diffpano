"""Offline checks for generation scope, prompt identity, and baseline settings."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import generation
from .baselines import specifications


class GenerationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.captions = {"state": "complete", "reviewed": True, "rows": []}
        self.dataset = {"rows": [], "second_seed_scene_ids": ["scene%02d" % index for index in range(8)]}
        for index in range(24):
            scene = "scene%02d" % index
            lines = ["Scene %d direction %d." % (index, direction) for direction in range(5)]
            path = self.root / (scene + ".txt")
            raw = ("\n".join(lines) + "\n").encode()
            path.write_bytes(raw)
            self.captions["rows"].append({"scene_id": scene, "prompt_id": "confirmation/" + scene,
                "source_image_sha256": "real-image-" + scene, "global_generation_description": "Global " + scene,
                "directional_descriptions": lines, "generation_prompt": {"path": str(path), "effective_sha256": hashlib.sha256(raw).hexdigest()}})
            self.dataset["rows"].append({"scene_id": scene, "image_sha256": "real-image-" + scene})

    def test_core_selects_exactly72_and_manifests_all152_unique_cases(self):
        rows = generation.case_definitions(self.captions, self.dataset)
        self.assertEqual(len(rows), 152)
        self.assertEqual(len({row["case_id"] for row in rows}), 152)
        selected = [row for row in rows if row["selected"]]
        self.assertEqual(len(selected), 72)
        self.assertEqual({row["method_id"] for row in selected}, {"sphere20", "select", "spherope"})
        self.assertTrue(all(row["seed"] == 0 for row in selected))
        self.assertEqual([row["method_id"] for row in rows[:3]], ["spherope", "sphere20", "select"])

    def test_extensions_are_explicit_and_seed_subset_is_stable(self):
        full = generation.case_definitions(self.captions, self.dataset, "full")
        self.assertEqual(sum(row["selected"] for row in full), 152)
        self.assertEqual({row["scene_id"] for row in full if row["tier"] == "seeds"}, set(self.dataset["second_seed_scene_ids"]))
        self.assertEqual(sum(row["tier"] == "generality" for row in full), 32)
        extra = generation.case_definitions(self.captions, self.dataset, "seeds")
        self.assertEqual(sum(row["selected"] for row in extra), 96)
        self.assertTrue(all(not row["selected"] for row in extra if row["tier"] in ("trained", "generality")))

    def test_caption_review_and_exact_prompt_bytes_are_required(self):
        self.captions["reviewed"] = False
        with self.assertRaisesRegex(ValueError, "reviewed"):
            generation.case_definitions(self.captions, self.dataset)
        self.captions["reviewed"] = True
        Path(self.captions["rows"][0]["generation_prompt"]["path"]).write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "bytes"):
            generation.case_definitions(self.captions, self.dataset)

    def test_scene_mismatch_or_unfrozen_seed_subset_is_rejected(self):
        changed = copy.deepcopy(self.captions)
        changed["rows"][0]["source_image_sha256"] = "another-image"
        with self.assertRaisesRegex(ValueError, "source"):
            generation.case_definitions(changed, self.dataset)
        self.dataset["second_seed_scene_ids"][0] = self.dataset["second_seed_scene_ids"][1]
        with self.assertRaisesRegex(ValueError, "Eight"):
            generation.case_definitions(self.captions, self.dataset)

    def test_spherope_quality_setting_is_not_public_fast_control(self):
        spec = specifications()["spherope"]
        self.assertEqual((spec["width"], spec["height"], spec["steps"]), (2048, 1024, 50))
        self.assertEqual(spec["public_fast_control"], {"width": 1024, "height": 512, "steps": 28, "selected": False})
        self.assertEqual(spec["checkpoint"], "black-forest-labs/FLUX.1-dev")
        self.assertTrue(spec["sfc_rope"] and spec["enable_circular_encoding"] and spec["semantic_distortion_cfg"])
        self.assertEqual(spec["semantic_distortion_gamma"], 6.)
        self.assertFalse(spec["enable_wrapped_noise"])

    def test_final_fallback_preserves_unrefined_select_science(self):
        spec = specifications()["select"]
        self.assertEqual((spec["steps"], spec["refinement"], spec["camera_count"]), (20, 0, 89))
        self.assertEqual((spec["gradient_mode"], spec["reference_mode"], spec["lambda_color"]), ("poisson_select", "screened", .1))
        self.assertEqual((spec["relative_tolerance"], spec["absolute_tolerance"], spec["max_iterations"]), (1e-5, 1e-7, 200))

    def test_retained_flux_and_sana_schedule_identity_preserves_infinite_bound(self):
        from .common import ROOT, read, digest
        from studies.tt_cea.common import historical_scheduler_json
        for backend in ("flux", "sana"):
            schedule = read(ROOT / ("outputs/gwtf-erp4k/20260923/" + backend + "/metadata.json"))["prepared_schedule"]
            normalized = historical_scheduler_json(schedule)
            self.assertEqual(generation.schedule_identity(schedule), digest(normalized))
            self.assertEqual(schedule["timesteps"], normalized["timesteps"])
            self.assertEqual(schedule["sigmas"], normalized["sigmas"])
            if backend == "sana":
                self.assertEqual(normalized["config"]["lambda_min_clipped"], "-Infinity")
                self.assertEqual(schedule["config"]["lambda_min_clipped"], float("-inf"))
        with self.assertRaisesRegex(ValueError, "outside historical scheduler sentinel"):
            generation.schedule_identity({"timesteps": [float("nan")]})

    def test_case_paths_cannot_escape_the_generation_directory(self):
        with self.assertRaises(AssertionError):
            generation.case_folder(self.root, {"case_id": "../outside"})
        with self.assertRaises(AssertionError):
            generation.case_folder(self.root, {"case_id": "/outside"})


if __name__ == "__main__":
    unittest.main()
