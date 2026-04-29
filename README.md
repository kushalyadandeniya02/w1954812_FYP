# Context-Aware Sentiment Analysis — Sarcasm and Gen Z Slang Detection in Social Media Text

This repository contains the full implementation for the Final Year Project (FYP) titled:  
**"A Context-Aware Machine Learning Model for Detecting Sarcasm and Gen Z Slangs in Youth-Driven Platforms."**

The project develops a complete end-to-end NLP pipeline that combines sarcasm detection, Gen Z slang identification, and sentiment analysis into a single Multi-Task Learning (MTL) model, enabling more accurate interpretation of youth-driven Reddit content.

---

## Project Aim

To develop a context-aware sentiment analysis pipeline using Machine Learning and Natural Language Processing that can accurately interpret sarcastic and slang-rich social media comments, enabling more reliable insights into Gen Z engagement and online sentiment.

---


## Dataset Overview

| Dataset | Description |
|---|---|
| `SAMPLE DATASET 50K.csv` | 50,000 Reddit comments with sarcasm labels, parent comment context, and metadata |
| `genz_slang.csv` | Gen Z slang terms with definitions |
| `all_slangs.csv` | Extended slang dictionary (merged and deduplicated at runtime) |

The primary dataset includes sarcastic and non-sarcastic comment pairs, making them well-suited for context-aware youth-focused sentiment analysis.

---

## Pipeline Overview (12 Steps)

### Step 1 — Setup & Imports
Installs all dependencies (`vaderSentiment`, `transformers`, `torch`, `rapidfuzz`, `wordcloud`, `imbalanced-learn`, `gradio`) and mounts Google Drive. Defines a `cache()` utility that pickles expensive computations to Drive, enabling seamless resumption across Colab sessions.

### Step 2 — Data Loading
Loads the three CSV datasets and validates column structure and dataset sizes.

### Step 3 — Preprocessing
Builds a unified slang dictionary from `genz_slang.csv` and `all_slangs.csv` (merged and deduplicated). Cleans the sarcasm dataset by dropping irrelevant columns (`author`, `ups`, `downs`, `date`), removing nulls and duplicates, and renaming `label` to `sarcasm_flag`. Constructs a `full_text` field by concatenating each comment with its parent comment to provide conversation context.

### Step 4 — Improved Slang Detection (v2)
Implements two slang detectors for direct comparison:

**v1 (Original — IPD baseline):** Exact dictionary lookup only.

**v2 (Improved — Final):** Three-layer detection system:
- **Layer 1:** O(1) set-based exact match
- **Layer 2:** Prefix-gated fuzzy matching via `rapidfuzz` (only fires when the 3-character prefix exists in the prefix index, reducing candidates to ~5)
- **Layer 3:** A single pre-compiled combined regex covering 16 multi-word Gen Z constructions (`no cap`, `fr fr`, `slay`, `low key`, `chef's kiss`, etc.)

Both detectors are cached to Drive to avoid recomputation.

### Step 5 — Improved Sentiment Labelling (v2 — RoBERTa)
Implements two sentiment labellers for comparison:

**v1 (Original — IPD baseline):** VADER compound score with polarity inversion for sarcastic comments (3 classes: Neutral / Negative / Positive).

**v2 (Improved — Final):** `cardiffnlp/twitter-roberta-base-sentiment-latest`, a model pre-trained on 124M tweets. Runs in batches of 1,024 with GPU acceleration and includes incremental partial-save every 10K rows for crash recovery. Agreement between VADER and RoBERTa is analysed overall and broken down by sarcasm flag to demonstrate the improvement on sarcastic comments specifically.

### Step 6 — Data Cleaning
Removes URLs, `@mentions`, strips `#` symbols while preserving the word (slang hashtags), collapses whitespace, removes comments shorter than 10 characters, and drops duplicates. Adds a `text_len` metadata feature (sarcastic comments tend to be shorter and punchier).

### Step 7 — Deep EDA (Exploratory Data Analysis)
Comprehensive visual analysis including:
- Sarcasm distribution and sentiment distribution (bar charts)
- Slang detection rate comparison (v1 vs v2)
- Correlation heatmap of sarcasm, slang, sentiment, and text length
- Word clouds for sarcastic vs non-sarcastic comments
- Top slang terms detected
- Sentiment distribution broken down by sarcasm flag
- Text length distribution by sarcasm and sentiment

### Step 8 — Dataset Split & TF-IDF Features
Stratified 70/15/15 train/validation/test split on the final dataset (using RoBERTa labels). TF-IDF vectorisation with 8,000 maximum features, combined via `scipy.sparse.hstack` with scaled metadata features (`text_len`, `sarcasm_flag`, `slang_flag`).

### Step 9 — Baseline ML Models
Trains and evaluates three traditional models on the TF-IDF feature matrix:
- **Logistic Regression**
- **Linear SVM**
- **Random Forest**

Each model is then re-trained using **SMOTE** oversampling to address class imbalance and compared on weighted F1 and accuracy. A target F1 line of 0.75 is shown on the comparison chart.

### Step 10 — Single-Task DistilBERT
Fine-tunes `distilbert-base-uncased` for 3-class sentiment classification using the Hugging Face `Trainer` API. Trained for 3 epochs with mixed-precision (fp16), cosine warmup, and best-model checkpointing. Tokenised datasets are cached to Drive. Evaluated on the held-out test set and results are added to the global comparison table.

### Step 11 — Multi-Task Learning (MTL) Model — *Main Novelty*
The core contribution of the project. A single shared `DistilBertModel` encoder feeds three independent task-specific heads simultaneously:

| Head | Task | Output |
|---|---|---|
| `sarcasm_head` | Sarcasm detection | Binary (2 classes) |
| `sentiment_head` | Sentiment classification | 3 classes (Neutral / Negative / Positive) |
| `slang_head` | Slang detection | Binary (2 classes) |

**Training:** Weighted combined loss (`α=1.0` sarcasm + `β=1.0` sentiment + `γ=0.5` slang) with AdamW (`lr=2e-5`), gradient clipping, and epoch-level checkpointing for crash recovery. Best model is selected based on sentiment F1. Training history (loss and F1 per epoch across all three tasks) is plotted.

**Inference:** A single forward pass produces all three outputs simultaneously.

### Step 12 — Full Model Comparison & Results
Generates a complete ranked comparison table and bar chart across all 8 models (3 baselines × 2 SMOTE variants + single-task DistilBERT + MTL DistilBERT). Also produces a dedicated MTL chart showing F1 scores across all three tasks. Final results table is saved to `model_results.csv` on Google Drive.

---

## Interactive Dashboard (Gradio)

A Gradio-powered dashboard is embedded at the end of the notebook and launches directly in Colab via a public share link.

**Features:**
- **Single Comment tab:** Enter any Reddit comment (with optional parent comment for context) and get instant sarcasm, sentiment, and slang predictions with emoji output.
- **Batch CSV upload tab:** Upload a CSV with a `comment` column (and optional `parent_comment` column) for bulk inference on up to 200 rows.
- Summary statistics (sarcasm rate, slang rate, sentiment breakdown) displayed per run.
- Pre-loaded example comments for quick testing.

The dashboard loads all model weights from Google Drive at startup — no retraining required.

---

## Technologies Used

| Category | Tools |
|---|---|
| Language | Python 3 |
| Environment | Google Colab (T4 GPU) |
| Data | Pandas, NumPy |
| NLP | NLTK (VADER), Hugging Face Transformers |
| Models | Scikit-learn, PyTorch, DistilBERT, Twitter-RoBERTa |
| Slang Matching | rapidfuzz |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| Visualisation | Matplotlib, Seaborn, WordCloud |
| Dashboard | Gradio |
| Caching | Python `pickle`, Google Drive |

---

## How to Run

1. Open the notebook in Google Colab and connect to a **GPU runtime** (Runtime → Change runtime type → T4 GPU).
2. Mount your Google Drive and update `BASE_DIR` in Step 1 to point to the folder containing the three CSV datasets.
3. Run all cells sequentially. The `cache()` utility will automatically skip expensive steps (slang detection, RoBERTa labelling, tokenisation, model training) if results are already saved to Drive.
4. The Gradio dashboard launches automatically at the end of Step 13 with a public share link.

---

## Key Design Decisions

**Context via parent comment concatenation:** Combining each reply with its parent comment gives the model the conversational context needed to interpret sarcasm correctly (e.g., "Yeah that went really well" is only sarcastic in context).

**RoBERTa over VADER for sentiment:** VADER is a lexical rule-based tool that struggles with irony and slang. Twitter-RoBERTa, trained on 124M tweets, better captures the register of Reddit comments and produces more reliable labels for model training.

**Multi-Task Learning as the primary novelty:** Instead of training three separate models, the MTL architecture shares a single DistilBERT encoder across sarcasm, sentiment, and slang tasks. The shared encoder learns representations that benefit all three tasks simultaneously, and a single forward pass produces all three outputs — making it efficient for real-time use in the Gradio dashboard.

**Prefix-gated fuzzy matching for slang:** Naively running fuzzy matching over the entire slang dictionary for every word would be prohibitively slow (O(n×m) comparisons). The prefix index reduces candidates to ~5 terms per word, making v2 slang detection practical at dataset scale.


## Student
**Kushalya Dandeniya** — w1954812  
BSc (Hons) Business Data Analytics, University of Westminster
