import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.ai.color_grading_engine import ColorGradingEngine
from src.ai import image_retrieval as image_retrieval_module
from src.ai.image_retrieval import ImageFeatureExtractor, ImageIndexDatabase
from src.ai.nlp_color_parser import ColorGradingParams
from src.ai.style_analyzer import StyleAnalyzer
from src.core.config_loader import load_llm_config
from src.utils.geometry_utils import GeometryUtils
from src.utils.image_io import imread, imwrite


class FakeClipModel:
    def encode(self, inputs):
        if isinstance(inputs, list):
            return np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (len(inputs), 1))
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)


class FakeFeatureExtractor:
    def __init__(self):
        self.model = None
        self._model_init_attempted = False
        self.init_calls = 0

    def _init_model(self):
        self.init_calls += 1
        self._model_init_attempted = True
        self.model = FakeClipModel()

    def _extract_dominant_colors(self, image, k=3):
        return [(10, 20, 30)] * k


class FakeSentenceTransformer:
    def __init__(self, model_path):
        self.model_path = model_path

    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, inputs):
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)


class CoreRegressionTests(unittest.TestCase):
    def test_llm_config_preserves_resource_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "llm_config.json"
            config_path.write_text(
                json.dumps({
                    "enabled": True,
                    "model_name": "local-model",
                    "device": "cuda",
                    "quantization": {"enabled": True, "bits": 4},
                    "max_memory": {"0": "8GB", "cpu": "16GB"},
                    "offload_folder": "./offload",
                    "trust_remote_code": True,
                }),
                encoding="utf-8",
            )

            config = load_llm_config(str(config_path))

        self.assertTrue(config["enabled"])
        self.assertEqual(config["quantization"]["bits"], 4)
        self.assertEqual(config["max_memory"]["0"], "8GB")
        self.assertEqual(config["offload_folder"], "./offload")
        self.assertTrue(config["trust_remote_code"])

    def test_unicode_image_io_roundtrip_and_invalid_extension(self):
        image = np.full((4, 5, 3), 127, dtype=np.uint8)
        with tempfile.TemporaryDirectory(prefix="图像测试_") as tmp:
            output_path = Path(tmp) / "中文路径.png"
            self.assertTrue(imwrite(str(output_path), image))
            loaded = imread(str(output_path))

            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.shape, image.shape)
            self.assertFalse(imwrite(str(Path(tmp) / "bad.unsupported"), image))

    def test_style_analyzer_lazily_prepares_clip_embeddings(self):
        extractor = FakeFeatureExtractor()
        analyzer = StyleAnalyzer(extractor)
        image = np.full((16, 16, 3), 128, dtype=np.uint8)

        content_tags, style_tags = analyzer._analyze_semantics_clip(image)

        self.assertEqual(extractor.init_calls, 1)
        self.assertTrue(content_tags)
        self.assertTrue(style_tags)

    def test_color_grading_output_contract(self):
        image = np.full((8, 8, 3), 128, dtype=np.uint8)
        params = ColorGradingParams(exposure=0.2, contrast=1.1, temperature=10, saturation=1.2)
        output = ColorGradingEngine().apply_grading(image, params)

        self.assertEqual(output.shape, image.shape)
        self.assertEqual(output.dtype, np.uint8)

    def test_grid_mesh_shape_and_first_faces(self):
        faces = GeometryUtils.create_grid_mesh(4, 3)

        self.assertEqual(faces.shape, (12, 3))
        self.assertEqual(faces.dtype, np.int32)
        self.assertEqual(faces[:2].tolist(), [[0, 4, 1], [1, 4, 5]])

    def test_clip_model_init_restores_offline_environment(self):
        original_cls = image_retrieval_module.SentenceTransformer
        original_hf = os.environ.get("HF_HUB_OFFLINE")
        original_transformers = os.environ.get("TRANSFORMERS_OFFLINE")
        image_retrieval_module.SentenceTransformer = FakeSentenceTransformer
        os.environ["HF_HUB_OFFLINE"] = "original"
        os.environ.pop("TRANSFORMERS_OFFLINE", None)

        try:
            with tempfile.TemporaryDirectory() as tmp:
                extractor = ImageFeatureExtractor(local_model_path=tmp)
                extractor._init_model()

            self.assertIsNotNone(extractor.model)
            self.assertEqual(os.environ.get("HF_HUB_OFFLINE"), "original")
            self.assertIsNone(os.environ.get("TRANSFORMERS_OFFLINE"))
        finally:
            image_retrieval_module.SentenceTransformer = original_cls
            if original_hf is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = original_hf
            if original_transformers is None:
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
            else:
                os.environ["TRANSFORMERS_OFFLINE"] = original_transformers

    def test_memory_index_filters_group_markers_consistently(self):
        db = object.__new__(ImageIndexDatabase)
        db.collection = None
        db.memory_index = [
            {
                "id": "__group__默认",
                "embedding": np.array([1.0, 0.0], dtype=np.float32),
                "metadata": {"path": "", "__is_group_marker__": "true"},
            },
            {
                "id": "empty-path",
                "embedding": np.array([1.0, 0.0], dtype=np.float32),
                "metadata": {"path": ""},
            },
            {
                "id": "image-a",
                "embedding": np.array([1.0, 0.0], dtype=np.float32),
                "metadata": {"path": "a.png"},
            },
            {
                "id": "image-b",
                "embedding": np.array([0.5, 0.5], dtype=np.float32),
                "metadata": {"path": "b.png"},
            },
        ]

        self.assertEqual(db.get_image_count(), 2)
        self.assertEqual([item["id"] for item in db.get_all_images(limit=10)], ["image-a", "image-b"])
        self.assertEqual(
            [item["id"] for item in db._memory_search(np.array([1.0, 0.0], dtype=np.float32), 10)],
            ["image-a", "image-b"],
        )


if __name__ == "__main__":
    unittest.main()
