## Unreleased

- Added API-backed LLM support for the natural-language color workflow, including OpenAI, Anthropic, and OpenAI-compatible chat-completions providers.
- Extended `llm_config.json` with explicit `provider` selection while preserving backward compatibility for existing local model configs.
- Replaced the object segmentation backend with SAM2 for point, box, and path selection workflows.
- Added SAM2 model download/configuration support and removed the vendored legacy segmentation source tree.
- Updated model documentation and dependency requirements for the SAM2-based workflow.
- Improved image-library text search with intent-aware color/background scoring, filename alias expansion, and metadata fallback recall for color-heavy queries.
- Tidied repository hygiene documentation around ignored local assets such as `data/`, `models/`, `artifacts/`, and Python cache directories.

## v1.2.0 - Product UI Platform Upgrade

- Added a modern command center, canvas HUD, processing overlay, drag/drop import affordances, and Ctrl+K command palette.
- Added a workflow panel for model readiness, asset context, next actions, and recent activity.
- Added custom Look Preset persistence and color-panel actions for saving, applying, and deleting reusable grading styles.
- Upgraded the image library, library manager, and image picker with summary metrics, clearer empty/error states, selection feedback, and consistent visual hierarchy.
- Refreshed the global PySide6 dark theme, font fallback strategy, and component styling across the application.
- Hardened AGI selection/export state handling and fixed selection overlay scaling regressions.
- Expanded automated regression coverage to 54 tests across core logic and PySide6 UI workflows.

## v1.1.0 - Product Optimization Pass

- Improved UI state handling so save, reset, undo, and compare actions match the current image context.
- Fixed programmatic color-panel resets triggering unintended neutral grading jobs.
- Added safer model/download handling, cooperative thread cancellation, and image I/O failure feedback.
- Hardened thumbnail sizing for extreme aspect-ratio images.
- Added core and UI regression tests.

## v1.0.0 - Initial Release

- Initial public release of AI Image Processor
- Core features:
  - Natural-language color grading
  - Image retrieval and indexing
  - 3D generation from images
- Desktop UI built with PySide6
