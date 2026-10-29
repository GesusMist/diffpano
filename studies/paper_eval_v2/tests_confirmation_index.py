"""Confirmation integration checks using tiny images, without models or API calls."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .common import read, write, sha
from . import confirmation_index as ci


class ConfirmationIndexTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.run = Path(temporary.name)
        self.prompt = self.run / "scene.txt"
        self.prompt.write_text("sky\nupper\nhorizon\nlower\nground\n")
        self.row = dict(index=0, case_id="confirmation24/scene/sphere20/seed0", scene_id="scene", prompt_id="fresh/scene",
            prompt_path=str(self.prompt), prompt_sha256=sha(self.prompt), directional_descriptions=self.prompt.read_text().splitlines(),
            method_id="sphere20", seed=0, tier="core", selected=True, backend="flux", semantic_sha256="semantic", caption_sha256="caption",
            source_image_sha256="real", specification=dict(width=8, height=4, steps=20, checkpoint="official", revision="pin", precision="bfloat16"))
        self.plan = dict(rows=[self.row], source_sha256="source")
        write(self.run / "generation/manifest.json", self.plan)

    def finished(self):
        from PIL import Image
        folder = ci.generation.case_folder(self.run, self.row)
        folder.mkdir(parents=True)
        Image.new("RGB", (8, 4), (30, 60, 90)).save(folder / "final.png")
        write(folder / "metadata.json", dict(semantic_sha256="semantic", generation_seconds=10))
        write(folder / "status.json", dict(state="complete", metadata_sha256=sha(folder / "metadata.json"), image_sha256=sha(folder / "final.png")))
        return folder

    def test_completed_image_and_metadata_are_verified(self):
        folder = self.finished()
        result = ci.records(self.run, self.plan)[0]
        self.assertTrue(result["eligible"])
        self.assertEqual(result["dimensions"], [8, 4])
        self.assertEqual(result["prompt"]["effective_sha256"], sha(self.prompt))
        write(folder / "metadata.json", dict(semantic_sha256="changed"))
        result = ci.records(self.run, self.plan)[0]
        self.assertFalse(result["eligible"])
        self.assertEqual(result["state"], "invalid")
        self.assertIsNone(result["image_sha256"])

    def test_pending_and_unselected_are_never_eligible(self):
        self.assertEqual(ci.records(self.run, self.plan)[0]["state"], "missing")
        self.row["selected"] = False
        self.assertEqual(ci.records(self.run, self.plan)[0]["state"], "not_selected")

    def cohort_fixture(self):
        old = dict(cohorts=dict(development24=dict(prompt_ids=["old"], matched={"primary": {"prompt_count": 1}}), confirmation24=dict(prompt_ids=[])),
                   prompts=[dict(prompt_id="old", memberships=["development24"])], selected_cells=[dict(prompt_id="old", case_id="old-case")])
        rows = []
        for i in range(24):
            for method in ci.METHODS:
                pid = "fresh/%02d" % i
                rows.append(dict(tier="core", prompt_id=pid, prompt=dict(prompt_id=pid, memberships=["confirmation24"]),
                    eligible=i == 0 and method != "spherope", seed=0, method_id=method, case_id=pid + "/" + method, panorama_id=pid + "/" + method))
        return old, rows

    def test_cohort_update_is_idempotent_and_preserves_development(self):
        old, rows = self.cohort_fixture()
        snapshot = copy.deepcopy(old)
        updated = ci.augment_cohorts(old, rows, "plan")
        self.assertEqual(old, snapshot)
        self.assertEqual(updated["cohorts"]["development24"], old["cohorts"]["development24"])
        self.assertEqual(updated["selected_cells"][0], old["selected_cells"][0])
        self.assertEqual(len(updated["selected_cells"]), 3)
        self.assertEqual(updated["cohorts"]["confirmation24"]["prompt_count"], 24)
        self.assertEqual(updated["cohorts"]["confirmation24"]["matched"]["primary"]["prompt_count"], 0)
        self.assertEqual(ci.augment_cohorts(updated, rows, "plan"), updated)

    def test_initial_snapshots_survive_incremental_updates(self):
        old, rows = self.cohort_fixture()
        write(self.run / "cohorts.json", old)
        write(self.run / "inventory/cohorts.json", old)
        with patch.object(ci.generation, "gate", return_value=self.plan), patch.object(ci, "records", return_value=[dict(row, selected=True, state="complete" if row["eligible"] else "missing") for row in rows]):
            ci.index(self.run)
            first = (self.run / "generation/initial_cohorts/root_cohorts.json").read_bytes()
            ci.index(self.run)
            self.assertEqual((self.run / "generation/initial_cohorts/root_cohorts.json").read_bytes(), first)
            self.assertEqual(read(self.run / "generation/initial_cohorts/root_cohorts.json"), old)

    def test_confirmation_pairs_use_only_core_and_explicit_common_prompts(self):
        rows = [dict(case_id=m + p, prompt_id=p, method_id=m, seed=0, metrics={"DS": value})
                for p in ("a", "b") for m, value in (("sphere20", 1), ("select", 3), ("spherope", 2))]
        rows.append(dict(case_id="extra-seed", prompt_id="a", method_id="select", seed=1, metrics={"DS": 99}))
        result = ci.summarize_confirmation(rows, ["a", "b", "missing"])
        pairs = result["paired_comparisons"]
        self.assertEqual([(p["baseline"], p["comparison"]) for p in pairs], list(ci.COMPARISONS))
        self.assertEqual([p["mean"] for p in pairs], [2, 1])
        self.assertTrue(all(p["prompt_count"] == 2 and p["missing_prompt_ids"] == ["missing"] for p in pairs))


if __name__ == "__main__":
    unittest.main()
