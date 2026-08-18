import unittest

from diffpano.pipelines import PIPELINE_REGISTRY


class PipelineRegistryTests(unittest.TestCase):
    def test_spherical_backends_are_registered_once(self):
        self.assertEqual(set(PIPELINE_REGISTRY), {"sana", "flux", "hunyuan_video", "ltx_video"})


if __name__ == "__main__":
    unittest.main()
