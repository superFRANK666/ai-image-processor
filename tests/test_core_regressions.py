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

    def test_color_params_normalize_rich_llm_schema(self):
        params = ColorGradingParams.from_dict({
            "hsl": {
                "blue": {"saturation": 28, "luminance": -8},
                "orange": {"lum": 12},
            },
            "color_wheels": {
                "shadows": {"hue": 195, "strength": 24},
                "highlights": {"hue": 38, "saturation": 20},
            },
            "cdl": {
                "slope": [1.05, 1.0, 0.96],
                "offset": [0.01, 0.0, -0.01],
                "power": [0.98, 1.0, 1.03],
                "saturation": 0.92,
            },
        })

        self.assertEqual(params.blue_saturation, 28)
        self.assertEqual(params.blue_luminance, -8)
        self.assertEqual(params.orange_luminance, 12)
        self.assertEqual(params.shadow_hue, 195)
        self.assertEqual(params.shadow_saturation, 24)
        self.assertEqual(params.highlight_hue, 38)
        self.assertEqual(params.highlight_saturation, 20)
        self.assertEqual(params.cdl_slope, [1.05, 1.0, 0.96])
        self.assertEqual(params.cdl_saturation, 0.92)

    def test_color_grading_rich_parameters_output_contract(self):
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        image[:, :8] = [190, 120, 40]
        image[:, 8:] = [40, 130, 220]
        params = ColorGradingParams(
            exposure=0.1,
            brightness=0.04,
            gamma=1.08,
            blue_saturation=30,
            orange_luminance=12,
            shadow_hue=195,
            shadow_saturation=25,
            highlight_hue=38,
            highlight_saturation=20,
            curve_darks=-12,
            curve_lights=10,
            texture=15,
            bloom=10,
            red_balance=6,
            cdl_slope=[1.02, 1.0, 0.98],
        )

        output = ColorGradingEngine().apply_grading(image, params)

        self.assertEqual(output.shape, image.shape)
        self.assertEqual(output.dtype, np.uint8)

    def test_traditional_parser_maps_semantics_to_rich_controls(self):
        from src.ai.nlp_color_parser import NLPColorParser

        parser = object.__new__(NLPColorParser)
        parser.text_encoder = None
        parser.use_llm = False
        parser.llm_analyzer = None

        sky = parser._traditional_parse("让天空更蓝更通透但肤色别太红", "让天空更蓝更通透但肤色别太红")
        cinematic = parser._traditional_parse("青橙电影感暗部冷高光暖", "青橙电影感暗部冷高光暖")

        self.assertGreater(sky.blue_saturation, 0)
        self.assertGreater(sky.dehaze, 0)
        self.assertLess(sky.red_saturation, 0)
        self.assertGreater(cinematic.shadow_saturation, 0)
        self.assertGreater(cinematic.highlight_saturation, 0)

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

    def test_look_preset_store_persists_updates_and_blocks_builtin_delete(self):
        from src.core.look_preset_store import LookPresetStore
        from src.ai.nlp_color_parser import ColorGradingParams

        with tempfile.TemporaryDirectory() as tmp:
            store_path = Path(tmp) / "looks.json"
            store = LookPresetStore(store_path)

            builtin = store.list_presets()[0]
            self.assertEqual(builtin.source, "builtin")
            self.assertFalse(store.delete_preset(builtin.id))

            saved = store.save_preset(
                "  Studio Look  ",
                ColorGradingParams(exposure=0.2, contrast=1.15),
                tags=["portrait", ""],
            )
            self.assertEqual(saved.name, "Studio Look")
            self.assertEqual(saved.source, "custom")
            self.assertTrue(store_path.exists())

            reloaded = LookPresetStore(store_path)
            loaded = reloaded.get_preset(saved.id)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.params["contrast"], 1.15)

            updated = reloaded.save_preset("Studio Look", {"contrast": 1.3})
            self.assertEqual(updated.id, saved.id)
            self.assertEqual(reloaded.get_preset(saved.id).params["contrast"], 1.3)

            self.assertTrue(reloaded.delete_preset(saved.id))
            self.assertIsNone(reloaded.get_preset(saved.id))


if __name__ == "__main__":
    unittest.main()
