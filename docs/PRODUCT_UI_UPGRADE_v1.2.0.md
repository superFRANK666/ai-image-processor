# Product UI Upgrade Notes (v1.2.0)

This release moves AI Image Processor from a feature-oriented desktop tool toward a more mature image workbench. The focus is not a cosmetic refresh alone: the UI now exposes product state, workflow intent, and asset context in places where users previously had to infer them from scattered buttons and status text.

## Product Goals

- Make the primary image workspace feel modern, calm, and operationally clear.
- Reduce ambiguity around model readiness, current asset state, selection state, and processing state.
- Turn image library flows into usable asset-management workflows instead of simple thumbnail lists.
- Create a UI architecture that can support future AI automation, batch workflows, and richer asset intelligence.

## Major Experience Changes

### Command Center and Canvas

- Added a top command center that summarizes the active asset, dimensions, history count, library count, and current status.
- Added direct commands for opening assets, intelligent grading, AGI camera workflows, image library focus, save, and compare.
- Added canvas HUD metadata, empty state actions, drag/drop image import, compare status, and processing overlay feedback.

### Workflow Panel

- Added a dedicated workbench tab that tracks file context, image size, history depth, library count, generated asset state, model readiness, workflow steps, next actions, and activity history.
- Model states now communicate loading, ready, and failed states in a consistent interface.

### Command Palette

- Added a Ctrl+K command palette with context-aware enabled/disabled commands.
- Command definitions are generated from current app state, so unavailable actions do not appear as silently broken controls.

### Look Presets

- Added persistent custom Look Presets for saving, applying, and deleting reusable grading styles.
- Built-in presets and custom presets share one UI flow while retaining source metadata for safe deletion behavior.

### Image Library

- Upgraded the side panel with a header, health badge, total/visible/group metrics, clearer search and action layout, and structured empty states.
- Empty, missing-file, no-result, and unavailable-database states now have distinct messages and safe actions.

### Library Manager

- Added a management header with total, loaded, and selected counts.
- The details panel now sits in a clearer management layout with preview and metadata states.
- Search/load failures remain visible in the UI instead of bubbling into fragile behavior.

### Image Picker

- Added a selection-decision header showing source, library availability/count, and selected count.
- The picker now synchronizes metrics for file browsing, library refresh, search results, multi-select, and single-select flows.

## Architecture Changes

- Added `src/ui/workflow_panel.py` for workflow metrics and model/activity state.
- Added `src/ui/command_palette.py` for command definitions and search-trigger behavior.
- Added `src/ui/font_utils.py` for Chinese-friendly UI font selection.
- Added `src/core/look_preset_store.py` for durable custom grading preset storage.
- Refactored `src/ui/style_sheet.py` into a broader application-level style system driven by object names and dynamic properties.

## Validation

Validated locally with:

```bash
python -m compileall -q main.py src tests
python -m unittest discover -s tests
git diff --check
```

The automated suite currently reports 54 passing tests.

Offscreen UI screenshots were generated during validation for the command center, command palette, processing overlay, image library panel, library manager, and image picker. These are local verification artifacts and are intentionally not committed.

## Follow-Up Opportunities

- Introduce a lightweight presenter/view-model layer to continue reducing `MainWindow` responsibilities.
- Add an automation queue for batch importing, indexing, grading, and exporting assets.
- Add persisted workspace/session state so the workbench can reopen the last active project context.
- Add richer command-palette search terms and future AI-assisted commands.
