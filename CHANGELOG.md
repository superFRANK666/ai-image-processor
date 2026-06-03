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
