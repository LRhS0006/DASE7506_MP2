# MP2: Comparing Candidates for Reliable Answer Selection

Frozen **14-feature candidate-comparison verifier**, DASE7506 MP2.

**Official test: 385/467 = 82.44111349036403%.** Full validation: 734/798 (91.98%); informative answer-disagreement validation: 291/355 (81.97%). The latter is the relevant difficulty-matched comparison. These denominators are not interchangeable.

The method first scores the supplied 1024-dimensional embeddings with logistic regression, then combines that score with answer-group quality, competitors, arithmetic contradictions and surface measurements. It outputs one candidate index per question; multiple candidates may be correct. No question is excluded from the official denominator.

## Package layout

- `method/`: final inference and feature calculations.
- `training/train.py`: train-only reproduction and optional nested cross-validation.
- `checkpoint/`: matching learned weights, scalers, configuration and asset hashes.
- `report/report.pdf`: final student-reviewed report.
- `evidence/`: official evaluation JSON, training settings, parameter check and experiment summary.
- `resources.json`: hardware, timing, dependencies and asset size.
- `AI_DISCLOSURE.md`, `LICENSE.txt`, `NOTICE.txt`: assistance and attribution.

## Install

Use Python 3.12. Obtain the complete original course release separately, leaving its data and tools unchanged. This repository does not duplicate the course dataset.

Run the following from this repository's root. Set `COURSE_CODE` to the original release's `code/` directory (absolute paths are recommended).

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-inference.txt
export COURSE_CODE="/absolute/path/to/DASE7506-MP2/code"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1
```

Install dependencies before offline evaluation. NumPy is sufficient for the final predictor. Training dependencies are separate; final inference requires no PyTorch.

## Reproduce the frozen checkpoint result, without retraining

```sh
python -B "$COURSE_CODE/predict.py" \
  --method method/submission.py --checkpoint checkpoint \
  --split test --output runs/test-predictions.json
python -B "$COURSE_CODE/score_predictions.py" \
  --split test --predictions runs/test-predictions.json \
  --output runs/test-score.json
```

Expected official `accuracy`: **82.44111349036403**, `correct`: **385**, `n`: **467**. The `accuracy` field already uses percent; do not multiply by 100 again.

The method does not assume bank length, inspect IDs, open references or load cached predictions. Four candidates per question is the official task contract.

## Reproduce training

Training is optional for score reproduction; use the released checkpoint above for peer review. To reproduce the final weights and the three primary difficult-training comparisons:

```sh
python -m pip install -r requirements-training.txt
python -B training/train.py --code-dir "$COURSE_CODE" \
  --output-dir runs/retrained --cross-validate
```

Expected internal difficult totals: independent regression **1672/2114**, 8-feature comparison **1680/2114**, final 14-feature method **1689/2114**. This command opens only training inputs and training references. It refuses to overwrite an existing output directory. Without `--cross-validate`, it reproduces final weights using the same five-fold OOF recipe, skipping nested comparisons.

The clean trainer was actually executed and all eight coefficient/scaler arrays matched the frozen checkpoint **exactly** on the recorded environment. It also produced the same NPZ hash. Training never replaces the release checkpoint. Numerical tolerances across different libraries/hardware should be distinguished from a score discrepancy.

## Protocol and limits

Protocol: `dase7506-mp2-open-hard-verifier-v5`. Only supplied training data fits weights/statistics. Validation was used repeatedly for development; test was evaluated only after the selected method was frozen. A later lossless export and one equality-check reproduction retained every frozen test choice; no test-based tuning took place.

No external datasets, pretrained model weights, online API or new LLM generation. Prediction is CPU-only. Total released inference-asset size, hashes, external runtime and known hardware are recorded in `resources.json`. Local timings do not certify the reference Xeon Platinum 8457C time limit. Ordinary code and supplied data are excluded from the 16 MiB asset budget; learned scalers, weights and configuration are included.

The original frozen bundle was exported after testing solely to remove unused research models and training-library dependencies. Arrays retain their original dtype and values. The base scaler preserves the original float32 intermediate rounding; the full 14-feature calculation and selection rule are unchanged. Provenance, training parameter identity and exact frozen prediction equality are recorded in `evidence/`.

## Author and submission

Sun Ruoheng, 3036707434. The final report is `report/report.pdf`.
Submit a fixed public code version and matching checkpoint download link.

## Attribution

See `AI_DISCLOSURE.md`, `LICENSE.txt` and `NOTICE.txt`.
