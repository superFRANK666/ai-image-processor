import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

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


class FakeNewApiSentenceTransformer:
    def __init__(self, model_path):
        self.model_path = model_path

    def get_embedding_dimension(self):
        return 3

    def get_sentence_embedding_dimension(self):
        raise AssertionError("legacy embedding dimension API should not be used")

    def encode(self, inputs):
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)


class FakeAPIResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload


class FakeAPISession:
    def __init__(self, payload):
        self.payload = payload
        self.requests = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.requests.append({
            "url": url,
            "headers": headers or {},
            "json": json or {},
            "timeout": timeout,
        })
        return FakeAPIResponse(self.payload)


class CoreRegressionTests(unittest.TestCase):
    def test_llm_config_preserves_resource_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "llm_config.json"
            config_path.write_text(
                json.dumps({
                    "enabled": True,
                    "provider": "local",
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
        self.assertEqual(config["provider"], "local")
        self.assertEqual(config["quantization"]["bits"], 4)
        self.assertEqual(config["max_memory"]["0"], "8GB")
        self.assertEqual(config["offload_folder"], "./offload")
        self.assertTrue(config["trust_remote_code"])

    def test_llm_config_supports_openai_compatible_api_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "llm_config.json"
            config_path.write_text(
                json.dumps({
                    "enabled": True,
                    "provider": "openai_compatible",
                    "model": "qwen2.5:7b",
                    "base_url": "http://localhost:11434/v1",
                    "api_key_required": False,
                    "temperature": "0.2",
                    "max_tokens": "256",
                    "extra_body": {"top_p": 0.9},
                }),
                encoding="utf-8",
            )

            config = load_llm_config(str(config_path))

        self.assertTrue(config["enabled"])
        self.assertEqual(config["provider"], "openai-compatible")
        self.assertEqual(config["model"], "qwen2.5:7b")
        self.assertEqual(config["base_url"], "http://localhost:11434/v1")
        self.assertFalse(config["api_key_required"])
        self.assertEqual(config["temperature"], 0.2)
        self.assertEqual(config["max_tokens"], 256)
        self.assertEqual(config["extra_body"]["top_p"], 0.9)

    def test_api_llm_analyzer_uses_openai_chat_completion_format(self):
        from src.ai.api_llm_analyzer import APILLMColorAnalyzer

        session = FakeAPISession({
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "is_color_related": True,
                        "reasoning": "需要暖色高光",
                        "parameters": {"temperature": 18, "highlight_saturation": 20},
                    })
                }
            }]
        })
        analyzer = APILLMColorAnalyzer(
            provider="openai",
            model="test-openai-model",
            api_key="test-key",
            session=session,
            timeout=12,
        )

        result = analyzer.analyze("夕阳电影感")

        self.assertTrue(result["is_color_related"])
        self.assertEqual(result["parameters"]["temperature"], 18)
        request = session.requests[0]
        self.assertEqual(request["url"], "https://api.openai.com/v1/chat/completions")
        self.assertEqual(request["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(request["json"]["model"], "test-openai-model")
        self.assertEqual(request["json"]["messages"][0]["role"], "system")
        self.assertEqual(request["json"]["messages"][1]["role"], "user")
        self.assertEqual(request["timeout"], 12)

    def test_api_llm_analyzer_uses_anthropic_messages_format(self):
        from src.ai.api_llm_analyzer import APILLMColorAnalyzer

        session = FakeAPISession({
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "is_color_related": True,
                    "reasoning": "蓝色通透",
                    "parameters": {"blue_saturation": 28, "dehaze": 16},
                }),
            }]
        })
        analyzer = APILLMColorAnalyzer(
            provider="anthropic",
            model="test-anthropic-model",
            api_key="anthropic-key",
            session=session,
        )

        result = analyzer.analyze("天空更蓝更通透")

        self.assertEqual(result["parameters"]["blue_saturation"], 28)
        request = session.requests[0]
        self.assertEqual(request["url"], "https://api.anthropic.com/v1/messages")
        self.assertEqual(request["headers"]["x-api-key"], "anthropic-key")
        self.assertIn("anthropic-version", request["headers"])
        self.assertEqual(request["json"]["model"], "test-anthropic-model")
        self.assertIn("system", request["json"])
        self.assertEqual(request["json"]["messages"][0]["role"], "user")

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

    def test_traditional_parser_maps_direct_color_words(self):
        from src.ai.nlp_color_parser import NLPColorParser

        parser = object.__new__(NLPColorParser)
        parser.text_encoder = None
        parser.use_llm = False
        parser.llm_analyzer = None

        golden = parser._traditional_parse("金黄色", "金黄色")
        deep_green = parser._traditional_parse("深绿色", "深绿色")

        self.assertGreater(golden.yellow_saturation, 0)
        self.assertGreater(golden.highlight_saturation, 0)
        self.assertGreater(deep_green.green_saturation, 0)
        self.assertLess(deep_green.green_luminance, 0)
        self.assertGreater(deep_green.midtone_saturation, 0)

    def test_async_llm_default_result_falls_back_to_traditional_color(self):
        from src.ai.nlp_color_parser import NLPColorParser

        class FakeAsyncLLM:
            def analyze_async(self, _text, on_success=None, on_error=None):
                if on_success:
                    on_success({
                        "is_color_related": True,
                        "reasoning": "模型识别为调色，但没有产出有效参数",
                        "parameters": {},
                    })

        parser = object.__new__(NLPColorParser)
        parser.text_encoder = None
        parser.use_llm = True
        parser.llm_analyzer = FakeAsyncLLM()
        results = []

        parser.parse_async("深绿色", on_success=results.append)

        self.assertEqual(len(results), 1)
        self.assertGreater(results[0].green_saturation, 0)
        self.assertLess(results[0].green_luminance, 0)

    def test_builtin_color_presets_fit_current_rich_parameter_schema(self):
        from src.core.config import COLOR_PRESETS

        valid_fields = set(ColorGradingParams.__dataclass_fields__)
        rich_fields = {
            "red_hue", "red_saturation", "red_luminance",
            "orange_hue", "orange_saturation", "orange_luminance",
            "yellow_hue", "yellow_saturation", "yellow_luminance",
            "green_hue", "green_saturation", "green_luminance",
            "aqua_hue", "aqua_saturation", "aqua_luminance",
            "blue_hue", "blue_saturation", "blue_luminance",
            "purple_hue", "purple_saturation", "purple_luminance",
            "magenta_hue", "magenta_saturation", "magenta_luminance",
            "shadow_hue", "shadow_saturation",
            "midtone_hue", "midtone_saturation",
            "highlight_hue", "highlight_saturation",
            "curve_shadows", "curve_darks", "curve_lights", "curve_highlights",
            "red_balance", "green_balance", "blue_balance",
            "cdl_slope", "cdl_offset", "cdl_power", "cdl_saturation",
            "texture", "midtone_detail", "dehaze", "bloom", "vignette", "grain", "fade",
        }

        for name, params in COLOR_PRESETS.items():
            with self.subTest(name=name):
                self.assertFalse(set(params) - valid_fields)
                self.assertTrue(set(params) & rich_fields)
                ColorGradingParams.from_dict(params)

    def test_grid_mesh_shape_and_first_faces(self):
        faces = GeometryUtils.create_grid_mesh(4, 3)

        self.assertEqual(faces.shape, (12, 3))
        self.assertEqual(faces.dtype, np.int32)
        self.assertEqual(faces[:2].tolist(), [[0, 4, 1], [1, 4, 5]])

    def test_model_download_specs_use_sam2_as_only_segmenter(self):
        from src.core.model_downloads import get_model_specs

        specs = get_model_specs()
        keys = {spec.key for spec in specs}

        self.assertIn("sam2_segmenter", keys)
        self.assertTrue(all("mobile" not in spec.key.casefold() for spec in specs))
        self.assertTrue(all("mobile" not in spec.role_title.casefold() for spec in specs))

    def test_incomplete_sam2_cache_is_not_treated_as_downloaded(self):
        from src.core.model_downloads import hf_snapshot_ready

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "sam2-hiera-tiny"
            cache_dir = model_dir / ".cache" / "huggingface"
            cache_dir.mkdir(parents=True)
            (cache_dir / "CACHEDIR.TAG").write_text("cache", encoding="utf-8")

            self.assertFalse(hf_snapshot_ready(model_dir))

            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            (model_dir / "preprocessor_config.json").write_text("{}", encoding="utf-8")
            (model_dir / "model.safetensors").write_bytes(b"weights")

            self.assertTrue(hf_snapshot_ready(model_dir))

    def test_model_cleanup_rejects_paths_outside_models_dir(self):
        from src.core.model_downloads import MODELS_DIR, clear_model_path

        with tempfile.TemporaryDirectory() as tmp:
            outside_model = Path(tmp) / "outside-model"
            outside_model.mkdir()

            with self.assertRaisesRegex(ValueError, "models"):
                clear_model_path(outside_model)

        with self.assertRaisesRegex(ValueError, "models"):
            clear_model_path(MODELS_DIR)

    def test_sam2_segmenter_selects_highest_scored_mask(self):
        from src.ai.agi_camera import ObjectSegmenter

        masks = np.zeros((1, 3, 4, 4), dtype=np.float32)
        masks[0, 0] = 0.1
        masks[0, 1] = 0.7
        masks[0, 2] = 0.3
        scores = np.array([[[0.2, 0.9, 0.4]]], dtype=np.float32)

        selected = ObjectSegmenter._select_best_mask([masks], scores)

        self.assertEqual(selected.shape, (4, 4))
        self.assertTrue(np.allclose(selected, 0.7))

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

    def test_clip_model_init_prefers_current_embedding_dimension_api(self):
        original_cls = image_retrieval_module.SentenceTransformer
        image_retrieval_module.SentenceTransformer = FakeNewApiSentenceTransformer

        try:
            with tempfile.TemporaryDirectory() as tmp:
                extractor = ImageFeatureExtractor(local_model_path=tmp)
                extractor._init_model()

            self.assertEqual(extractor.embedding_dim, 3)
            self.assertIs(extractor.image_model, extractor.model)
            self.assertEqual(extractor.image_embedding_backend, "clip_image")
        finally:
            image_retrieval_module.SentenceTransformer = original_cls

    def test_text_search_skips_semantic_without_clip_image_encoder(self):
        class FakeCollection:
            def __init__(self):
                self.query_calls = 0

            def get(self, include=None):
                return {
                    "ids": ["image-a"],
                    "metadatas": [{
                        "path": "white-id-photo.png",
                        "dominant_colors": json.dumps([[255, 255, 255]]),
                        "brightness": 250,
                        "contrast": 10,
                    }],
                }

            def query(self, *args, **kwargs):
                self.query_calls += 1
                return {"ids": [[]], "metadatas": [[]], "distances": [[]]}

        db = object.__new__(ImageIndexDatabase)
        db.collection = FakeCollection()
        db.feature_extractor = SimpleNamespace(
            model=FakeClipModel(),
            image_model=None,
            _model_init_attempted=True,
            _is_multilingual=True,
        )
        db._text_index_backend_checked = False

        results = ImageIndexDatabase.search_by_text(db, "复刻去年海边旅行的蓝色色调", top_k=1)

        self.assertEqual(results, [])
        self.assertEqual(db.collection.query_calls, 0)

    def test_text_search_filters_legacy_embedding_candidates(self):
        legacy_metadata = {
            "path": "white-id-photo.png",
            "dominant_colors": json.dumps([[255, 255, 255]]),
            "brightness": 250,
            "contrast": 10,
        }

        class FakeCollection:
            def __init__(self):
                self.query_calls = 0

            def get(self, include=None):
                return {"ids": ["image-a"], "metadatas": [legacy_metadata]}

            def query(self, *args, **kwargs):
                self.query_calls += 1
                return {
                    "ids": [["image-a"]],
                    "metadatas": [[legacy_metadata]],
                    "distances": [[0.01]],
                    "embeddings": [[[1.0, 0.0, 0.0]]],
                }

        db = object.__new__(ImageIndexDatabase)
        db.collection = FakeCollection()
        db.feature_extractor = SimpleNamespace(
            model=FakeClipModel(),
            image_model=object(),
            _model_init_attempted=True,
            _is_multilingual=True,
        )
        db._text_index_backend_checked = False
        db.rebuild_all_indexes = lambda: 0

        results = ImageIndexDatabase.search_by_text(db, "复刻去年海边旅行的蓝色色调", top_k=1)

        self.assertEqual(results, [])
        self.assertEqual(db.collection.query_calls, 1)

    def test_color_match_uses_hsv_and_dominant_rank(self):
        db = object.__new__(ImageIndexDatabase)
        blue_background = {
            "dominant_colors": json.dumps([
                [197, 118, 68],
                [25, 27, 28],
                [53, 72, 98],
            ])
        }
        student_card = {
            "dominant_colors": json.dumps([
                [253, 253, 253],
                [64, 50, 52],
                [12, 11, 11],
                [170, 170, 180],
                [29, 30, 209],
            ])
        }

        blue_score = db._compute_color_match_score(blue_background, ["blue"])
        student_score = db._compute_color_match_score(student_card, ["blue"])

        self.assertGreater(blue_score, 0.8)
        self.assertLess(student_score, 0.15)

    def test_color_only_query_prioritizes_actual_color_over_texture(self):
        db = object.__new__(ImageIndexDatabase)
        candidates = [
            {
                "id": "student-card",
                "path": "C:/tmp/学生证.jpg",
                "semantic_similarity": 0.214,
                "metadata": {
                    "path": "C:/tmp/学生证.jpg",
                    "dominant_colors": json.dumps([
                        [253, 253, 253],
                        [64, 50, 52],
                        [12, 11, 11],
                        [170, 170, 180],
                        [29, 30, 209],
                    ]),
                    "brightness": 208.7,
                    "contrast": 84.0,
                },
            },
            {
                "id": "blue-id-photo",
                "path": "C:/tmp/蓝底证件照.jpg",
                "semantic_similarity": 0.215,
                "metadata": {
                    "path": "C:/tmp/蓝底证件照.jpg",
                    "dominant_colors": json.dumps([
                        [197, 118, 68],
                        [25, 27, 28],
                        [53, 72, 98],
                        [107, 129, 156],
                    ]),
                    "brightness": 96.2,
                    "contrast": 46.0,
                },
            },
        ]

        ranked = db._multi_feature_rerank(candidates, "蓝色", np.array([], dtype=np.float32))

        self.assertEqual(ranked[0]["id"], "blue-id-photo")
        self.assertGreater(ranked[0]["final_score"], ranked[1]["final_score"] + 0.5)

    def test_color_name_search_expands_background_aliases(self):
        class FakeCollection:
            def get(self, include=None):
                return {
                    "ids": ["blue", "red"],
                    "metadatas": [
                        {"path": "C:/tmp/蓝底证件照.jpg"},
                        {"path": "C:/tmp/红底证件照.jpg"},
                    ],
                }

        db = object.__new__(ImageIndexDatabase)
        db.collection = FakeCollection()

        results = db.search_by_name("蓝色证件照", top_k=5)

        self.assertEqual([item["id"] for item in results], ["blue"])
        self.assertGreaterEqual(results[0]["similarity"], 0.9)

    def test_text_search_merges_color_metadata_candidates(self):
        student_metadata = {
            "path": "C:/tmp/student-card.jpg",
            "embedding_backend": "clip_image",
            "dominant_colors": json.dumps([
                [253, 253, 253],
                [64, 50, 52],
                [12, 11, 11],
            ]),
            "brightness": 208.7,
            "contrast": 84.0,
        }
        blue_metadata = {
            "path": "C:/tmp/plain-reference.jpg",
            "embedding_backend": "clip_image",
            "dominant_colors": json.dumps([
                [197, 118, 68],
                [25, 27, 28],
            ]),
            "brightness": 96.2,
            "contrast": 46.0,
        }

        class FakeCollection:
            def get(self, include=None):
                return {
                    "ids": ["student", "blue"],
                    "metadatas": [student_metadata, blue_metadata],
                }

            def query(self, *args, **kwargs):
                return {
                    "ids": [["student"]],
                    "metadatas": [[student_metadata]],
                    "distances": [[0.786]],
                    "embeddings": [[[1.0, 0.0, 0.0]]],
                }

        db = object.__new__(ImageIndexDatabase)
        db.collection = FakeCollection()
        db.feature_extractor = SimpleNamespace(
            model=FakeClipModel(),
            image_model=object(),
            _model_init_attempted=True,
            _is_multilingual=True,
        )
        db._text_index_backend_checked = False

        results = ImageIndexDatabase.search_by_text(db, "蓝色", top_k=1)

        self.assertEqual([item["id"] for item in results], ["blue"])
        self.assertEqual(results[0]["match_type"], "metadata_color")

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
