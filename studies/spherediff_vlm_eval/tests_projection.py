"""CPU geometry, published-prompt integrity and read-only cache regressions."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from studies.panorama_metrics.render import render
from . import preprocessing
from .protocol import (CAMERAS, CRITERIA, PROMPT_PATH, PROMPT_PROVENANCE_PATH, PROMPT_SHA256,
                       VIEWS, protocol_record, prompt_text, rendering_protocol_record)


def direction_erp(height=256):
    latitude = (.5 - (np.arange(height) + .5) / height) * np.pi
    longitude = ((np.arange(height * 2) + .5) / (height * 2) - .5) * 2 * np.pi
    lon, lat = np.meshgrid(longitude, latitude)
    vectors = np.stack((np.sin(lon) * np.cos(lat), np.sin(lat), np.cos(lon) * np.cos(lat)), -1)
    return np.rint((vectors + 1) * 127.5).astype(np.uint8)


class ProjectionTests(unittest.TestCase):
    def test_fourteen_cameras_and_512_outputs(self):
        expected = tuple((float(y), float(p)) for p in (-45, 0, 45) for y in (0, 90, 180, 270))
        self.assertEqual(VIEWS, expected + ((0., 90.), (0., -90.)))
        self.assertEqual(len(set(c['view_id'] for c in CAMERAS)), 14)
        views = render(direction_erp(), VIEWS, size=512)
        self.assertEqual(views.shape, (14, 512, 512, 3))
        for (yaw, pitch), view in zip(VIEWS, views):
            y, p = np.deg2rad([yaw, pitch])
            forward = np.array((np.cos(p) * np.sin(y), np.sin(p), np.cos(p) * np.cos(y)))
            actual = view[255:257, 255:257].mean(axis=(0, 1)) / 127.5 - 1
            np.testing.assert_allclose(actual, forward, atol=.012)
        self.assertEqual(sum(c['is_seam'] for c in CAMERAS), 3)
        self.assertTrue(all(c['yaw'] == 180 and not c['is_polar'] for c in CAMERAS if c['is_seam']))

    def test_polar_orientation(self):
        north, south = render(direction_erp(), ((0., 90.), (0., -90.)), size=512)
        self.assertGreater(north[256, 256, 1], 250)
        self.assertLess(south[256, 256, 1], 5)
        # Fixed yaw0: right is +x at both poles. North up=-z; south up=+z.
        for view in (north, south):
            self.assertGreater(int(view[256, 448, 0]), int(view[256, 64, 0]) + 100)
        self.assertLess(int(north[64, 256, 2]), int(north[448, 256, 2]) - 100)
        self.assertGreater(int(south[64, 256, 2]), int(south[448, 256, 2]) + 100)

    def test_seam_at_180_crosses_the_image_center(self):
        erp = np.zeros((256, 512, 3), dtype=np.uint8)
        erp[:, :8] = 255
        erp[:, -8:] = 255
        seam, ordinary = render(erp, ((180., 0.), (0., 0.)), size=512)
        self.assertTrue(np.all(seam[256, 252:260] > 245))
        self.assertTrue(np.all(ordinary[256, 252:260] == 0))
        # Distinguish the two original boundaries: left output samples ERP right.
        erp[:, -8:] = (0, 0, 255)
        erp[:, :8] = (255, 0, 0)
        seam = render(erp, ((180., 0.),), size=512)[0]
        self.assertGreater(int(seam[256, 250, 2]), 245)
        self.assertGreater(int(seam[256, 261, 0]), 245)

    def test_periodic_longitude(self):
        erp = direction_erp()
        views = render(erp, ((180., 0.), (-180., 0.), (0., 0.), (360., 0.)), size=512)
        np.testing.assert_array_equal(views[0], views[1])
        np.testing.assert_array_equal(views[2], views[3])
        seam = views[0].astype(float)
        self.assertLess(np.abs(seam[:, 255] - seam[:, 256]).max(), 3)

    def test_prompt_integrity_and_published_provenance(self):
        self.assertEqual(hashlib.sha256(PROMPT_PATH.read_bytes()).hexdigest(), PROMPT_SHA256)
        provenance = json.loads(PROMPT_PROVENANCE_PATH.read_text())
        self.assertEqual(provenance['asset_sha256'], PROMPT_SHA256)
        self.assertEqual(provenance['verification_status'], 'exact_published_wording_verified')
        self.assertEqual(provenance['prompt_pdf_page'], 17)
        self.assertEqual(provenance['preprocessing_pdf_page'], 19)
        self.assertEqual(provenance['license_url'], 'https://creativecommons.org/licenses/by/4.0/')
        text = prompt_text()
        self.assertEqual(text.count('{image}'), 1)
        for criterion in CRITERIA:
            self.assertIn(f'- {criterion}: (Brief reason) → Score', text)
        self.assertIn('Excellent (5), Good (4), Fair (3), Poor (2), and Awful (1).', text)
        self.assertEqual(protocol_record()['rendering']['width'], 512)
        with tempfile.TemporaryDirectory() as folder:
            bad = Path(folder) / 'changed.txt'
            bad.write_text(text + 'altered')
            with patch('studies.spherediff_vlm_eval.protocol.PROMPT_PATH', bad):
                with self.assertRaisesRegex(ValueError, 'integrity'):
                    prompt_text()

    def test_cache_resume_invalidation_and_historical_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'historical' / 'final.png'
            source.parent.mkdir()
            Image.fromarray(direction_erp()).save(source)
            before = source.read_bytes()
            stat_before = source.stat()
            record = dict(panorama_id='case-a', method_id='rgb', prompt_id='full-prompt-hash',
                          prompt_name='scene', image_path=str(source),
                          image_sha256=hashlib.sha256(before).hexdigest())
            output = root / 'new-evaluation'
            first = preprocessing.preprocess([record], output)
            self.assertEqual(len(first), 14)
            mtimes = {row['view_path']: Path(row['view_path']).stat().st_mtime_ns for row in first}
            with patch('studies.panorama_metrics.render.render', side_effect=AssertionError('cache miss')):
                second = preprocessing.preprocess([record], output)
            self.assertEqual(first, second)
            self.assertEqual(mtimes, {path: Path(path).stat().st_mtime_ns for path in mtimes})
            corrupted = Path(first[0]['view_path'])
            corrupted.write_bytes(b'corrupt png')
            repaired = preprocessing.preprocess([record], output)
            self.assertEqual(first, repaired)
            changed_protocol = rendering_protocol_record()
            changed_protocol['test_cache_revision'] = 2
            with patch.object(preprocessing, 'rendering_protocol_record', return_value=changed_protocol):
                changed = preprocessing.render_panorama(record, output)
            self.assertNotEqual(changed[0]['view_path'], first[0]['view_path'])
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(source.stat().st_mtime_ns, stat_before.st_mtime_ns)
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                preprocessing.render_panorama(dict(record, image_sha256='0' * 64), output)
            with self.assertRaisesRegex(ValueError, 'separate'):
                preprocessing.render_panorama(record, source.parent)
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                preprocessing.preprocess([record, record], output)


if __name__ == '__main__':
    unittest.main()
