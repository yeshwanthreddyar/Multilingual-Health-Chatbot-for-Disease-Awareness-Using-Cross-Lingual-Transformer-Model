## Dataset Overview

HealthBot trains its intent and disease ensembles on a **multilingual conversational dataset**
constructed from three primary sources plus chat logs:

- **Symptom2Disease** (`Symptom2Disease.csv`)
- **PubMedQA** (`data/pubmedqa/ori_pqau.json`)
- **IndicNLG** (Hindi and other Indian‑language medical/exam text)
- **Online logs** from the running system (`train_from_logs.py`)

The unified dataset is materialised as:

- `data/processed/combined_dataset.json` – created by `train.py`.
- Optional per‑language splits under `data/processed/{lang}_dataset.json`.

The target scale is **15k+ samples** across English and key Indic languages (hi, bn, te, ta, mr, gu,
kn, ml, pa, or, as).

## Canonical Sample Schema

Across loaders, rows are normalised to the following **canonical schema** (fields may be missing
for some sources but are typed consistently):

- **`text`** (`str`, required): user‑style utterance or question.
- **`intent`** (`str`, required): mapped to one of the classifier intents:
  - `disease_information`, `symptom_reporting`, `prevention_guidance`,
    `vaccination_schedule`, `emergency_assessment`, `general_health_query`,
    plus a few intermediate labels such as `treatment_info`, `diagnosis_info`, `cause_info`
    that are collapsed via `map_intent_to_classifier_format` in `train.py`.
- **`language`** (`str`, required): ISO‑style language code (e.g. `en`, `hi`, `bn`, `te`).
- **`split`** (`str`, optional): one of `train`, `dev`, `test` (or omitted for logs).
- **`symptoms`** (`List[str]`, optional): keyword‑level symptom mentions.
- **`disease` / `diseases`** (`str` / `List[str]`, optional): dataset‑specific disease labels.
- **`is_emergency`** (`bool`, optional): heuristic flag for emergency triage.
- **`answer` / `context`** (`str`, optional): long‑form answers or supporting text (mainly PubMedQA/IndicNLG).
- **`source`** (`str`, required): original dataset identifier (e.g. `symptom2disease`, `pubmedqa_ori_pqau`, `indicnlg_hi`).

Scripts that need a compact view for intent classification (e.g. `train_from_logs.py`) typically
retain just `["text", "intent", "language"]`.

## Loaders and Preprocessing

- **Symptom2Disease** – `app/data/load_symptom2disease.py`:
  - Reads `Symptom2Disease.csv` into the canonical format:
    - `text`: free‑text symptom description.
    - `disease`: raw label, later mapped to classifier labels in `train_disease_classifier.py`.
    - `language`: `en`.
    - `intent`: `symptom_reporting`.
    - `symptoms`: extracted via `extract_symptoms_from_text`.
    - `is_emergency`: keyword‑based triage via `check_emergency_symptoms`.
- **PubMedQA** – `app/data/load_pubmedqa.py`:
  - Converts the original QA structure into HealthBot samples:
    - `text`: question.
    - `context`: supporting context.
    - `answer`: long answer.
    - `language`: `en`.
    - `intent`: via `classify_pubmed_intent(question, context)`.
    - `symptoms` / `diseases` / `is_emergency`: heuristic extractors.
- **IndicNLG** – `app/data/load_indicnlg.py`:
  - Supports both language‑specific `{lang}.{split}.json` and generic `{split}.json` layouts.
  - `convert_indicnlg_format` maps each record to HealthBot fields:
    - `text`, `answer`, `language`, `intent`, `symptoms`, `is_emergency`, `split`, `source`.
  - Coverage includes Hindi, Bengali, Telugu, Tamil, Marathi, Gujarati, Kannada, Malayalam, Punjabi,
    Odia, Assamese, etc.
- **Chat logs** – `train_from_logs.py`:
  - Pulls health‑related messages from `app.models.log.Log`, normalises their intents via `INTENT_MAP`,
    and merges them with `combined_dataset.json` to keep the classifier aligned with real user traffic.

## Combined Dataset Construction

- **`train.py`** (root):
  - `prepare_combined_dataset()`:
    - Loads Symptom2Disease, PubMedQA, and IndicNLG through the loader modules.
    - Normalises them into the canonical schema and concatenates them into a single DataFrame.
  - `create_language_specific_datasets(df)`:
    - Splits `combined_df` by `language` and saves JSON files under `data/processed/`.
  - `train_models_on_combined(df)`:
    - Generates 384‑dim IndicBERT/mBERT embeddings via `get_embedding_generator()`.
    - Trains the intent ensemble and reports both overall and per‑language accuracy.

For downstream tasks, `app/data/__init__.py` exposes a helper:

- `load_combined_conversations(path="data/processed/combined_dataset.json")`:
  - Returns a pandas DataFrame with the canonical schema, ready to be fed into the ML pipeline
    or evaluation scripts.

## Reproducibility and Splits

- Train/validation/test splits are created in `train.py` using `sklearn.model_selection.train_test_split`
  with `random_state=42` and stratification when feasible.
- The `split` column from IndicNLG and Symptom2Disease is preserved; PubMedQA is currently used mostly
  for training/evaluation rather than strict production splitting.
- Future work can include exporting explicit indices for each split to a small manifest file to enable
  byte‑for‑byte reproducible experiments.

## Dataset Properties and Statistics

Basic dataset properties are computed programmatically and can be regenerated whenever the raw data
changes:

- **Helper**: `app/data/__init__.py`
  - `load_combined_conversations(path="data/processed/combined_dataset.json")` loads the unified
    dataset and projects it to the canonical schema.
  - `summarize_dataset(df)` returns aggregate counts:
    - `total_samples`
    - `languages` → per‑language sample counts
    - `intents` → per‑intent sample counts
    - `by_language_and_intent` → nested breakdown for monitoring multilingual balance.
- **Typical workflow**:
  1. Run `python train.py` to regenerate `data/processed/combined_dataset.json` from all sources.
  2. Open a short Python session or notebook and call the helpers:

     ```python
     from app.data import load_combined_conversations, summarize_dataset

     df = load_combined_conversations()
     stats = summarize_dataset(df)
     print(stats)
     ```

This summary is used during experimentation to verify that the dataset maintains the intended
scale (15k+ samples) and a healthy distribution across English and key Indic languages.

### Implementation Mapping (Paper → Code)

- **Multilingual conversational corpus (15k+ samples)** → loaders in `app/data/load_symptom2disease.py`,
  `app/data/load_pubmedqa.py`, `app/data/load_indicnlg.py`, combined in `train.py`.
- **Dataset schema and preprocessing** → canonical fields and heuristics documented above, implemented
  across the loader modules and `train.py`.
- **Train/eval splits and metrics** → splitting and accuracy computation in `train.py` and
  `evaluation/evaluate.py`.

### Research Paper Mapping Cheat Sheet

- **Paper: Dataset Construction (15k+ Multilingual Samples)** → concrete file locations and loaders for
  Symptom2Disease, PubMedQA, IndicNLG, and logs; unified into `data/processed/combined_dataset.json` by `train.py`.
- **Paper: Canonical Conversation Schema** → the `text`, `intent`, `language`, `split`, `symptoms`, `disease(s)`,
  `is_emergency`, `answer/context`, and `source` fields described here and produced by the loader modules.
- **Paper: Preprocessing & Normalisation** → heuristic mappers such as `map_intent_to_classifier_format` in `train.py`
  and the per-dataset converters in `app/data/*.py`.
- **Paper: Train/Validation/Test Protocol** → `train.py` and `evaluation/evaluate.py`, which define splits,
  random seeds, and per-language metrics that back the reported results.

