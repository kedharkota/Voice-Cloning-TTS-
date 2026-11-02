# Reproducibility and visual sources

## Scope

This guide describes what is available in the committed project and what is required to reproduce it. The documentation update inspected source and saved outputs; it did not rerun model training or hardware/simulation experiments.

## Required inputs

Training audio, manifests, generated YAML files under `configs/`, and trained checkpoints are not included in the visible root tree. Notebook 08 reads those configuration files immediately. The four root `.py` files are empty placeholders, so the notebooks are the implementation entry point. Model downloads may require their own access and license acceptance. Use recordings with the speaker's permission.

## Entry point

Read `PROJECT_DOCUMENTATION.md`, then run the environment notebook and data pipeline before training component notebooks. Use a compatible CUDA/PyTorch environment and consult each notebook's installation cells for its model-specific requirements.

Open the notebooks through Jupyter and run them in order. The integration notebook launches the Gradio interface after the models and configuration files are available.

## Evidence and interpretation

The documentation lists target metrics and an evaluation framework for ASR, translation, speaker similarity, emotion, and speech quality. Target values are not verified achieved results. Reproducible pipeline metrics, checkpoints, and listening samples are not committed as standalone artifacts in the root tree.

## Visual provenance

All screenshots and experiment figures are tied to repository material. Workflow SVGs are explanatory illustrations. The full image inventory is in [assets/sources.json](assets/sources.json).

| Visual | Source |
| --- | --- |
| [overview.svg](assets/overview.svg) | New explanatory project-workflow diagram based on the committed source and documentation; not a screenshot or measured result. |

## Validation

```bash
python tools/check_docs.py
```

The standard-library checker validates README/docs local link destinations and SVG/PNG/JPEG headers. It makes no external requests and installs no project runtime dependencies. External web links, model accuracy, image semantics, and full application execution need separate review.

## Refresh notebook visuals

The exporter reads saved `image/png` outputs without executing notebook code:

```bash
python tools/export_notebook_images.py "08_pipeline_integration_demo.ipynb"
```

Review the exported images, remove any unsuitable or private content, choose the figures that explain the project, and update the README links and `docs/assets/sources.json`. It does not manufacture results when a notebook has no saved PNG output.

## Reproduction record

For a new experiment or demo, record the source commit, environment versions, data provenance and split, configuration, command/notebook order, and resulting artifacts. Only label a measurement as reproduced after completing that run. Keep credentials and restricted datasets outside version control.
