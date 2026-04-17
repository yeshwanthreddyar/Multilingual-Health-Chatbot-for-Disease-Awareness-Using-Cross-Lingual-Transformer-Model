## Model Architecture

### Embeddings: IndicBERT + Multilingual MiniLM (mBERT family)

- **Primary encoder (`EmbeddingGenerator`, `app/nlp/embeddings.py`)**:
  - Uses `ai4bharat/indic-bert` for Indic scripts by default and `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` as multilingual fallback.
  - All encoder outputs are **projected/padded to a fixed `EMBEDDING_DIM = 384`**, ensuring a single representation size across:
    - Training (`train.py`, `train_from_logs.py`, `train_disease_classifier.py`)
    - Runtime inference (`NLPPipeline`, `MLPipeline`)
- **Language routing**:
  - `detect_language` (`app/nlp/language_detection.py`) detects ISO 639-1 language codes for messages.
  - `NLPPipeline` (`app/nlp/pipeline.py`) calls `EmbeddingGenerator.encode_single(text, lang_hint=lang)` so IndicBERT is preferred for Indic scripts and MiniLM/mBERT is preferred for English.

### Intent Classification Ensemble

- **Intent classifier (`IntentClassifier`, `app/ml/intent_classifier.py`)**:
  - Inputs: 384‑dim embeddings from `EmbeddingGenerator`.
  - Outputs: probabilities over **six intent classes**:
    - `disease_information`, `symptom_reporting`, `prevention_guidance`,
      `vaccination_schedule`, `emergency_assessment`, `general_health_query`.
  - Architecture: **ensemble of three classical models**:
    - Gaussian Naive Bayes
    - Random Forest (100 trees)
    - Gradient Boosting (50 estimators)
  - The three probability distributions are combined via a **weighted average**; `predict_single` exposes both the winning intent and its confidence.
- **Training**:
  - `train.py` builds a multilingual conversational dataset (Symptom2Disease + PubMedQA + IndicNLG) and trains the ensemble.
  - `train_from_logs.py` fine‑tunes/retrains from `Log` records stored in the database, again using the same 384‑dim embeddings.
  - Trained models are stored under `models/intent_classifier_*.pkl` and loaded at runtime in `get_ml_pipeline()`.

### Disease Classification Ensemble

- **Disease classifier (`DiseaseClassifier`, `app/ml/disease_classifier.py`)**:
  - Labels aligned with the knowledge graph disease keys (`DISEASE_LABELS` matching `DISEASE_GRAPH` in `app/knowledge_base/graph.py`).
  - Architecture: **GaussianNB + RandomForest + GradientBoosting ensemble**, mirroring the intent classifier design.
  - Input features:
    - In the online pipeline: **multi‑hot symptom vectors** over canonical ICD‑like symptom codes (`CANONICAL_SYMPTOM_CODES` in `MLPipeline`).
    - In training: 384‑dim symptom embeddings derived from Symptom2Disease text via `EmbeddingGenerator`, plus multilingual conversational embeddings where disease labels are available.
  - Output: **top‑k diseases with confidence scores**, used strictly for **advisory** (not diagnostic) messages.
- **Training**:
  - `train_disease_classifier.py` converts Symptom2Disease into disease labels aligned with the graph KB, trains the ensemble, and saves it as `models/disease_classifier_trained.pkl`.
  - `train.py` additionally trains a multilingual conversational disease classifier from the combined dataset (where disease labels are present), persisting it as `models/disease_classifier_trained_multilingual.pkl` alongside distribution stats.
  - At runtime, `get_ml_pipeline()` prefers the multilingual disease model when available and only falls back to the Symptom2Disease‑only model or a small synthetic fit when no trained models are available.

### Knowledge Graph and Medical Knowledge Base

- **Disease graph (`app/knowledge_base/graph.py`)**:
  - In‑memory graph keyed by disease codes (e.g. `dengue`, `typhoid`, `malaria`) with:
    - Node type `type = "disease"`.
    - ICD‑style `who_code`.
    - Symptom code lists, prevention tips, and vaccination guidelines.
    - Multilingual names via a `names` map (`en`, `hi`, `bn`, `te`, etc.) and helpers to compute localised rich context for ML top‑k predictions.
  - Helper APIs:
    - `get_disease_info`, `get_disease_name(lang=...)`
    - `get_diseases_by_symptoms(symptom_codes)` for KB‑driven ranking.
    - `get_rich_context_for_topk(topk)` and `get_localised_rich_context_for_topk(topk, lang)` to enrich ML outputs with KB metadata.
- **Retrieval layer (`app/knowledge_base/retriever.py`)**:
  - `retrieve_disease_advisory(disease_key, lang)` returns:
    - Localised disease name.
    - Prevention tips, vaccination options.
    - Any time‑varying updates from `app.data.load_health_updates`.
  - `retrieve_for_symptoms(symptom_codes, lang)` maps raw symptom codes to top‑k disease advisories.

### Dialogue and LLM Orchestration

- **Dialogue Manager (`app/dialog/manager.py`)**:
  - Consumes outputs from `MLPipeline`:
    - Intent, symptom list, top‑k diseases, emergency flag.
  - Decides high‑level **action**:
    - `emergency`, `location`, `prevention`, `vaccination`, `follow_up`, `general`.
  - Connects to:
    - **Knowledge base** via `retrieve_disease_advisory(...)` and `retrieve_for_symptoms(...)`.
    - **Location service** for nearby hospitals (`app/integrations/location_service.py`).
    - **Outbreak/alert system** indirectly via alert push to subscribers (`main.py`, `app/outbreak/detector.py`, `app/integrations/government_mock.py`).
- **LLM (`app/llm/ollama_client.py`, `run_ollama_chat.py`)**:
  - Used **only for phrasing, summarisation, and general health education**:
    - Response phrasing for advice/prevention.
    - General Q&A when no structured disease/symptom match is found.
  - All **factual content** (diseases, prevention, vaccination, official advisories) is **grounded** in:
    - Ensemble predictions (intent + disease).
    - The knowledge graph.
    - External health feeds (via `app/integrations/health_portals.py` and outbreak detection).

### Implementation Mapping (Paper → Code)

- **Transformer-based multilingual embeddings** → `app/nlp/embeddings.py`, `app/nlp/pipeline.py`.
- **Ensemble intent classifier** → `app/ml/intent_classifier.py`, training in `train.py`, `train_from_logs.py`.
- **Ensemble disease classifier** → `app/ml/disease_classifier.py`, training in `train_disease_classifier.py`.
- **Graph-based knowledge representation** → `app/knowledge_base/graph.py`, access via `app/knowledge_base/retriever.py`.
- **Knowledge-guided conversational flow** → `app/ml/pipeline.py` + `app/dialog/manager.py` orchestrating ML, KB, and LLM.

### Research Paper Mapping Cheat Sheet

- **Paper: Multilingual Transformer Embedding Layer** → IndicBERT + multilingual MiniLM/mBERT, implemented via
  `EmbeddingGenerator` and `NLPPipeline` (sections “Multilingual Embeddings” and “End-to-End ML Pipeline” above).
- **Paper: Intent Recognition over Embeddings** → `IntentClassifier` ensemble and its training/inference paths,
  wired into `MLPipeline.run(...)`.
- **Paper: Disease Prediction using Symptom Features** → `DiseaseClassifier` ensemble, `CANONICAL_SYMPTOM_CODES`,
  `_symptom_vector(...)`, and the disease top-k outputs in `MLPipeline`.
- **Paper: Graph-based Medical Knowledge Base** → `DISEASE_GRAPH` and helpers in `app/knowledge_base/graph.py`
  plus enrichment functions in `app/knowledge_base/retriever.py` and `get_rich_context_for_topk(...)`.
- **Paper: Knowledge-Guided Conversational Agent** → `DialogueManager` decision logic and the way it consumes
  ML + KB outputs and optionally uses the LLM (`app/llm/ollama_client.py`) only for phrasing.

## Model Architecture – HealthBot

This document links the research paper architecture to the current implementation.

### 1. Multilingual Embeddings (IndicBERT + mBERT)

- **Implementation**: `app/nlp/embeddings.py`, `app/nlp/pipeline.py`
- **Models**:
  - Primary: **IndicBERT** – `INDICBERT_MODEL = "ai4bharat/indic-bert"`
  - Fallback: **mBERT-like multilingual model** – `MBERT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"`
- **Runtime flow**:
  1. `NLPPipeline.process(text)` (`app/nlp/pipeline.py`):
     - Detects language via `detect_language(text)`.
     - Tokenizes via `tokenize(text, lang)`.
     - Calls `EmbeddingGenerator.encode_single(text, lang_hint=lang)`.
  2. `EmbeddingGenerator` (`app/nlp/embeddings.py`):
     - Tries IndicBERT first for non‑English or unknown languages.
     - Falls back to the multilingual MiniLM/mBERT model if IndicBERT is unavailable or fails.
     - As a last resort (for tests without models), returns a small random vector with fixed dimension 384.

At **inference time**, all downstream ML components receive embeddings through this single entry point, ensuring a consistent representation across the system.

### 2. Intent Recognition – Ensemble over Transformer Embeddings

- **Implementation**:
  - Training: `train.py`, `train_from_logs.py`
  - Runtime: `app/ml/intent_classifier.py`, `app/ml/pipeline.py`
- **Architecture**:
  - Input features: sentence embeddings from `EmbeddingGenerator` (one vector per utterance).
  - Classes (`INTENT_CLASSES`):  
    `disease_information`, `symptom_reporting`, `prevention_guidance`, `vaccination_schedule`, `emergency_assessment`, `general_health_query`.
  - Ensemble classifier (`IntentClassifier`):
    - Gaussian Naive Bayes
    - Random Forest (100 trees)
    - Gradient Boosting (50 estimators)
    - Predictions are a **weighted average** of the three models’ probabilities.

**Training pipeline** (`train.py`):
- Loads conversational datasets (Symptom2Disease, PubMedQA, IndicNLG).
- Generates embeddings using `get_embedding_generator().encode_single(...)` with language hints.
- Trains `IntentClassifier` on these embeddings and saves it to `models/intent_classifier_trained.pkl`.

**Online retraining from logs** (`train_from_logs.py`):
- Reads health‑related chat logs from the DB.
- Uses `get_embedding_generator()` to embed user messages.
- Retrains `IntentClassifier` and saves `models/intent_classifier_trained_from_logs.pkl`.

**Inference** (`app/ml/pipeline.py`):
- Takes the 1D embedding from `NLPPipeline`.
- Pads or truncates to a fixed dimension 384 to match the fitted classifier.
- Calls `IntentClassifier.predict_single(...)` to get `(intent, confidence)`.

### 3. Disease Classification – Ensemble over Symptom Features

- **Implementation**:
  - Training: `train_disease_classifier.py`
  - Runtime: `app/ml/disease_classifier.py`, `app/ml/pipeline.py`
- **Architecture**:
  - Labels (`DISEASE_LABELS`): mapped to high‑level disease keys such as `common_cold`, `flu`, `dengue`, `malaria`, `typhoid`, `respiratory_infection`, etc.
  - Classifier (`DiseaseClassifier`):
    - Gaussian Naive Bayes
    - Random Forest
    - Gradient Boosting
    - Weighted probability ensemble (mirrors the intent classifier design).

Two complementary feature paths are supported:

1. **Symptom multi‑hot vectors (default runtime path)** – `app/ml/pipeline.py`:
   - Symptoms are extracted by `extract_symptoms(text, tokens)` and mapped to canonical codes.
   - `_symptom_vector(...)` builds a multi‑hot vector over `CANONICAL_SYMPTOM_CODES`.
   - `DiseaseClassifier.top_k(...)` returns top‑k diseases and confidence scores.

2. **Symptom text embeddings (training path)** – `train_disease_classifier.py`:
   - Symptom descriptions from Symptom2Disease are embedded using `EmbeddingGenerator`.
   - These embeddings are used to train the same `DiseaseClassifier` ensemble.

The combination of (1) rule‑based symptom extraction plus (2) learned mapping from symptom features to disease labels aligns with the paper’s description of P(disease | symptoms) using ensemble classifiers.

### 4. End‑to‑End ML Pipeline

- **Implementation**: `app/ml/pipeline.py`
- **Steps for a single user utterance**:
  1. `NLPPipeline.process(text)` → language, tokens, embedding.
  2. `MLPipeline.run(...)`:
     - Intent prediction from the embedding via `IntentClassifier`.
     - Symptom extraction and emergency flag.
     - Disease top‑k prediction from symptom features via `DiseaseClassifier`.
  3. Outputs: intent, confidence, list of symptoms, top‑3 diseases, emergency flag, and an `advisory_only` marker.

These outputs are then consumed by the **Dialogue Manager** (`app/dialog/manager.py`) and the **Knowledge Base** (`app/knowledge_base/graph.py`, `app/knowledge_base/retriever.py`) to generate final, safety‑aware responses, matching the research paper’s architecture (transformer embeddings → ensemble classifiers → graph‑based knowledge → response generation).

