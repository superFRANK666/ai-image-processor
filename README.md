# AI Image Processor (v1.2.0)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

AI Image Processor is a desktop application for local AI-powered image workflows, including:
- Natural-language color grading and reusable Look Presets
- Image library indexing, semantic retrieval, and visual asset governance
- Single-image and object-focused 3D generation
- A modern PySide6 workbench with command center, canvas HUD, and command palette

The project is designed for local/offline usage after model download, with a PySide6 GUI and modular AI backends.

## Key Features

- Natural language color grading with rule-based parsing and optional local LLM analysis
- Custom Look Presets for saving, applying, and managing repeatable grading styles
- Local image library management with semantic + visual feature search
- Product-grade asset workflows: library health summaries, picker metrics, empty states, and manager detail panels
- Modern command center, workflow panel, and Ctrl+K command palette for fast navigation
- Canvas HUD with asset metadata, drag/drop import, compare state, empty state, and processing overlay
- 3D mesh and animation generation from images (depth + segmentation workflow)
- Object selection with MobileSAM (point/box/path interactions)
- Chinese-path-safe image I/O utilities
- Lazy/asynchronous model loading to reduce UI startup blocking

## Tech Stack

- Language: Python 3.9+
- GUI: PySide6 (Qt for Python)
- Imaging: OpenCV, Pillow, scikit-image, NumPy
- AI/ML: PyTorch, Transformers, Sentence-Transformers, ONNX Runtime, MobileSAM
- Retrieval/Storage: ChromaDB
- 3D: Open3D, trimesh

## Project Structure

```text
AIImageProcessor/
├── main.py
├── requirements.txt
├── pyproject.toml
├── llm_config.example.json
├── run.bat
├── run.sh
├── src/
│   ├── ai/
│   ├── core/
│   ├── mobile_sam/
│   ├── ui/
│   └── utils/
├── scripts/
│   ├── setup.bat
│   ├── setup.sh
│   └── download_all_models.py
├── docs/
└── submission/
```

## Setup and Installation

### 1. Create environment

Windows:
```powershell
python -m venv venv
venv\Scripts\activate
```

Linux/macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Optional GPU ONNX runtime:
- Replace `onnxruntime` with `onnxruntime-gpu` in your environment.

### 3. Download required models

```bash
python scripts/download_all_models.py
```

Model weights are intentionally not tracked in Git.

## Run

Recommended:
```bash
python main.py
```

Helper launchers:
- Windows: `run.bat`
- Linux/macOS: `bash run.sh`

Useful flags:
```bash
python main.py --check-deps
python main.py --debug
```

## LLM Configuration (Optional)

To enable local LLM analysis for color commands:
1. Copy `llm_config.example.json` to `llm_config.json`
2. Select one valid config block and simplify it to a single active config object
3. Ensure corresponding model files are available locally

If `llm_config.json` is absent or disabled, the app falls back to non-LLM parsing.

## Testing and Validation

Recommended validation commands:
```bash
python -m compileall -q main.py src scripts
python -m unittest discover -s tests
python main.py --check-deps
```

The v1.2.0 UI/product pass includes automated core and PySide6 UI regression tests. The current suite covers:
- color panel reset and Look Preset behavior
- image viewer HUD, processing overlay, and canvas state
- command center and command palette state
- image library, picker, and manager selection/empty/error states
- AGI camera selection/export state regressions

## Build / Packaging Notes

- Python packaging metadata is defined in `pyproject.toml`.
- The console entry point is `ai-image-processor = main:main`.
- For GitHub publication, do not commit:
  - virtual environments
  - model files under `models/`
  - local config files such as `llm_config.json`

## Documentation

Additional docs are available under `docs/`, including:
- LLM quickstart and configuration guides
- v1.2.0 product UI upgrade notes
- performance and optimization notes
- code review notes from previous iterations

## License

MIT License. See [LICENSE](./LICENSE).

For vendored third-party code notices, see [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md).

