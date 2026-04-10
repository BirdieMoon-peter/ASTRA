# ASTRA Experiment Source Code

ASTRA is an experimental research codebase for studying **strategic optimism decomposition** in Chinese analyst reports. The project separates report-level fundamental sentiment from rhetorical optimism using a constrained, evidence-grounded pipeline, then evaluates the resulting signals with NLP metrics and downstream cross-sectional finance diagnostics.

## What is included

This repository contains the upload-safe source package:

- `src/astra/` — ASTRA source modules
  - `decomposition/` — structured report decomposition
  - `neutralization/` — counterfactual-neutral rewriting utilities
  - `verification/` — factual-preservation and evidence checks
  - `scoring/` — report-level signal construction
  - `evaluation/` — NLP, calibration, human-eval, and cost-analysis utilities
  - `finance/` — finance signal construction and cross-sectional backtest logic
  - `pipelines/` — runnable pipeline entry points
- `scripts/` — helper scripts for annotation, evaluation, finance search, and leakage checks
- `tests/` — unit and contract tests
- `configs/` — sanitized experiment/backtest configuration files

## What is intentionally excluded

To avoid leaking private or non-uploadable material, this repository intentionally excludes:

- raw data and processed datasets
- generated outputs, predictions, figures, and paper artifacts
- local runtime configuration under `.local/`
- API keys, tokens, credentials, and private endpoint configuration
- internal planning notes and project-local documents under `docs/`
- local cache files such as `__pycache__/` and `.DS_Store`

The checked-in config files are sanitized. Any real LLM provider keys or private endpoints should be supplied only through local environment variables or an ignored `.local/llm_config.json` file.

## Basic setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
```

Install the Python packages required by the modules you plan to run. The code is organized as a source-layout package, so commands should usually be run with:

```bash
PYTHONPATH=src python3 -m astra.pipelines.run_nlp_eval
```

or, for tests:

```bash
PYTHONPATH=src python3 -m pytest tests
```

## LLM configuration

ASTRA's LLM client reads settings from environment variables or from `.local/llm_config.json`.

Example local config, not committed:

```json
{
  "provider": "openai_compatible",
  "model": "your-model-name",
  "base_url": "https://your-private-endpoint.example/v1",
  "api_key": "YOUR_API_KEY",
  "max_tokens": 4000
}
```

Equivalent environment variables are also supported, such as:

```bash
export LLM_PROVIDER=openai_compatible
export OPENAI_MODEL=your-model-name
export OPENAI_BASE_URL=https://your-private-endpoint.example/v1
export OPENAI_API_KEY=YOUR_API_KEY
```

Do not commit real secrets.

## Finance backtest settings

The finance backtest configuration is documented in:

```text
configs/finance_backtest_protocol.json
```

It specifies the point-in-time alignment, tradability filters, cross-sectional normalization, neutralization controls, rolling protocol, and transaction-cost stress settings used by the research backtest code.

## Reproducibility note

This repository contains source code and sanitized configuration only. To reproduce full paper-facing numbers, provide the corresponding data package and generated prediction artifacts in the expected local paths. These files are excluded from git by design.
