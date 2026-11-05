# IndicClaimVer@FIRE 2026
### Claim Verification and Evidence Retrieval for Indian Languages

<p align="center">
  <a href="https://sites.google.com/view/indicclaimver2026/home">
    <img src="https://img.shields.io/badge/Shared%20Task-FIRE%202026-blue" alt="FIRE 2026">
  </a>
  <img src="https://img.shields.io/badge/Domain-Multilingual%20NLP-green" alt="Multilingual NLP">
  <img src="https://img.shields.io/badge/Task-Claim%20Verification-orange" alt="Claim Verification">
  <img src="https://img.shields.io/badge/Framework-PyTorch-red" alt="PyTorch">
  <img src="https://img.shields.io/badge/Models-Transformers-yellow" alt="Transformers">
</p>

**IndicClaimVer@FIRE 2026** is a shared task on multilingual claim verification and evidence retrieval for Indian languages, organized as part of the 18th Forum for Information Retrieval Evaluation (FIRE 2026).

This repository presents our implementation of an end-to-end multilingual claim verification framework addressing both evidence selection and veracity prediction in a closed-evidence setting and end-to-end claim verification in an open-evidence setting.

The system combines multilingual dense retrieval, sparse lexical retrieval, hybrid ranking, cross-encoder reranking, diversity-aware evidence selection, transformer-based claim classification, ensemble learning, and retrieval-augmented generation (RAG).

**Team:** TokenX

**Shared Task:** IndicClaimVer@FIRE 2026

**Repository:** https://github.com/rachanabn20/IndicClaimVer-FIRE2026

**Official Website:** https://sites.google.com/view/indicclaimver2026/home

---

## Table of Contents

- [1. Overview](#1-overview)
- [2. Shared Task Description](#2-shared-task-description)
- [3. Proposed System Architecture](#3-proposed-system-architecture)
- [4. Subtask 1: Evidence Selection and Veracity Prediction](#4-subtask-1-evidence-selection-and-veracity-prediction)
- [5. Subtask 2: End-to-End Claim Verification](#5-subtask-2-end-to-end-claim-verification)
- [6. Models and Technologies](#6-models-and-technologies)
- [7. Experimental Configuration](#7-experimental-configuration)
- [8. Dataset Organization](#8-dataset-organization)
- [9. Installation](#9-installation)
- [10. Execution](#10-execution)
- [11. Output and Submission Files](#11-output-and-submission-files)
- [12. Evaluation Metrics](#12-evaluation-metrics)
- [13. Implementation Considerations](#13-implementation-considerations)
- [14. Limitations and Future Improvements](#14-limitations-and-future-improvements)
- [15. Acknowledgements](#15-acknowledgements)
- [16. Citation](#16-citation)

---

## 1. Overview

The increasing circulation of misinformation across multilingual digital platforms creates significant challenges for automated fact-checking systems. These challenges become more pronounced in multilingual environments where claims are expressed in different languages, scripts, and code-mixed forms.

IndicClaimVer@FIRE 2026 addresses these challenges by evaluating claim verification and evidence retrieval systems across four language settings:

- English
- Hindi
- Bengali
- Code-mixed text

Our implementation follows a retrieval-based verification framework in which relevant evidence is retrieved for a given claim and subsequently used to determine its veracity.

The proposed system consists of two complementary pipelines:

1. **Closed-Evidence Verification:** Retrieves relevant evidence from a provided evidence collection and predicts whether the claim is supported or refuted.
2. **Open-Evidence Verification:** Retrieves evidence, generates an evidence summary, predicts claim veracity, and produces a natural-language justification.

### Key Components

- Multilingual text preprocessing and normalization.
- Dense semantic retrieval using multilingual sentence embedding models.
- Sparse lexical retrieval using BM25.
- TF-IDF-based corpus representation.
- Hybrid retrieval using weighted score fusion and Reciprocal Rank Fusion (RRF).
- Near-duplicate evidence filtering.
- Cross-encoder-based evidence reranking.
- Maximal Marginal Relevance (MMR) for evidence diversity.
- Multilingual transformer-based veracity classification.
- Focal loss and class-weighted learning.
- Majority, soft, and weighted ensemble voting.
- Retrieval-augmented evidence summarization and justification generation.
- Automated submission validation and JSON generation.

---

## 2. Shared Task Description

IndicClaimVer@FIRE 2026 consists of two subtasks.

### Subtask 1: Evidence Selection + Veracity Prediction

**Setting:** Closed Evidence

Given a claim and a predefined collection of candidate evidence, the objective is to identify the most relevant evidence and determine whether the claim is supported or refuted.

The system performs two primary operations:

1. Evidence retrieval and ranking.
2. Evidence-conditioned veracity classification.

The implementation retrieves the top three evidence passages and uses them as contextual input to a multilingual transformer classifier.

### Subtask 2: End-to-End Claim Verification

**Setting:** Open Evidence

Given a claim without a predefined gold evidence passage, the system retrieves relevant information from the configured evidence collection and generates:

1. An evidence summary.
2. A veracity prediction.
3. A natural-language justification.

The implementation uses a retrieval-augmented generation pipeline with multilingual evidence retrieval and an mT5-based text-generation component, with an extractive fallback when the generator cannot be loaded.

The official task description specifies evidence summaries of approximately 100–150 words and justifications of approximately 100–120 words. The official label terminology and submission schema should be followed when preparing final submissions.

---

## 3. Proposed System Architecture

The overall framework consists of a shared preprocessing and retrieval component followed by task-specific verification pipelines.

### 3.1 Overall Workflow

```text
                    INPUT CLAIM
                         |
                         v
                TEXT PREPROCESSING
                         |
                         v
                EVIDENCE RETRIEVAL
                         |
             +-----------+-----------+
             |                       |
             v                       v
       DENSE RETRIEVAL          SPARSE RETRIEVAL
       Sentence Embeddings          BM25
       FAISS Similarity             Lexical Matching
             |                       |
             +-----------+-----------+
                         |
                         v
                  SCORE FUSION
               Weighted Fusion / RRF
                         |
                         v
               NEAR-DUPLICATE FILTER
                         |
                         v
                CROSS-ENCODER
                   RERANKING
                         |
                         v
                MMR DIVERSIFICATION
                         |
                  TOP EVIDENCE
                         |
              +----------+----------+
              |                     |
              v                     v
          SUBTASK 1              SUBTASK 2
       CLOSED EVIDENCE          OPEN EVIDENCE
              |                     |
              v                     v
       CLAIM-EVIDENCE           EVIDENCE
       CLASSIFICATION           SUMMARIZATION
              |                     |
              v                     v
       TRANSFORMER              VERACITY
       CLASSIFIERS              VERIFICATION
              |                     |
              v                     v
       ENSEMBLE VOTING          JUSTIFICATION
              |                     |
              v                     v
       SUPPORTS / REFUTES       STRUCTURED OUTPUT
              |                     |
              +----------+----------+
                         |
                         v
                  JSON SUBMISSIONS
```

### 3.2 Common Preprocessing

The preprocessing pipeline applies the following operations:

- HTML entity decoding.
- HTML tag removal.
- URL removal.
- Emoji removal.
- Unicode NFKC normalization.
- Repeated punctuation normalization.
- Whitespace normalization.
- Empty-text filtering.
- Duplicate-record removal.

The same cleaning function is applied to claims and evidence passages to maintain consistent text representations during retrieval and classification.

---

## 4. Subtask 1: Evidence Selection and Veracity Prediction

The Subtask 1 pipeline integrates hybrid evidence retrieval with transformer-based claim classification.

### 4.1 Evidence Pool Construction

The system loads the provided training and test evidence collections.

Evidence records are normalized to a common structure containing:

- Evidence identifier.
- Evidence text.
- Language metadata, when available.

The evidence pool is represented using an identifier-to-text mapping, allowing retrieved passages to be associated with their original evidence identifiers.

When the training evidence pool contains insufficient records, the implementation includes a fallback that constructs a pool from unique gold evidence passages in the training and development datasets.

For test-time retrieval, a dedicated test evidence pool is preferred when it contains sufficient records.

### 4.2 Dense Semantic Retrieval

Dense retrieval captures semantic similarity between claims and evidence passages, including cases where lexical overlap is limited.

The implementation evaluates the following multilingual embedding models:

| Model | Description |
|---|---|
| `BAAI/bge-m3` | Multilingual dense embedding model |
| `intfloat/multilingual-e5-base` | Multilingual text embedding model |
| `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | Multilingual sentence embedding model |

Evidence passages are encoded into dense vector representations.

FAISS is used to construct an inner-product similarity index. Both evidence and query embeddings are L2-normalized, making inner-product similarity equivalent to cosine similarity.

The embedding model with the highest development-sample NDCG@3 is selected for subsequent retrieval experiments.

### 4.3 Sparse Lexical Retrieval

BM25 is employed to identify evidence passages with strong lexical overlap with the input claim.

The evidence collection is tokenized using whitespace-based tokenization and indexed using `BM25Okapi`.

In addition, TF-IDF representations are constructed using unigrams and bigrams with a maximum vocabulary size of 50,000 features.

The implemented retrieval ranking uses BM25 alongside dense retrieval; TF-IDF is prepared as an additional lexical corpus representation.

### 4.4 Hybrid Evidence Retrieval

The system combines dense and sparse retrieval results using two fusion strategies.

**Weighted Score Fusion**

Dense and BM25 scores are independently normalized using min-max normalization.

The combined score is calculated as:

\[
S(e)=\alpha S_{\mathrm{dense}}(e)
+(1-\alpha)S_{\mathrm{BM25}}(e)
\]

where:

- \(S_{\mathrm{dense}}\) represents the normalized dense similarity score.
- \(S_{\mathrm{BM25}}\) represents the normalized BM25 score.
- \(\alpha\) controls the contribution of dense retrieval.

The implementation evaluates multiple values of \(\alpha\) and selects the value producing the highest NDCG@3 on the sampled development data.

**Reciprocal Rank Fusion (RRF)**

The second strategy combines the ranking positions produced by dense and sparse retrieval.

\[
\operatorname{RRF}(e)=
\frac{1}{k+r_{\mathrm{dense}}(e)}
+
\frac{1}{k+r_{\mathrm{BM25}}(e)}
\]

where \(k\) is the RRF constant, configured as 60, and \(r(e)\) denotes the zero-based rank of an evidence passage.

Both strategies are compared using retrieval evaluation, and the better-performing fusion strategy is selected.

### 4.5 Near-Duplicate Evidence Filtering

To reduce redundant evidence, the system applies a near-duplicate filtering procedure using `SequenceMatcher`.

Candidate passages are compared using their textual similarity.

Passages with similarity above the configured threshold of 0.85 are filtered, retaining a more diverse candidate collection.

### 4.6 Cross-Encoder Reranking

The retrieved candidates are reranked using:

`cross-encoder/ms-marco-MiniLM-L-12-v2`

The cross-encoder evaluates claim-evidence pairs jointly and produces relevance scores.

The top-ranked passages are retained for subsequent evidence selection.

### 4.7 Maximal Marginal Relevance

Maximal Marginal Relevance (MMR) is applied to balance evidence relevance and diversity.

The selection objective is:

\[
\operatorname{MMR}(e)=
\lambda \operatorname{Rel}(e)
-(1-\lambda)\operatorname{Red}(e,S)
\]

where:

- \(\operatorname{Rel}(e)\) is the reranker relevance score.
- \(\operatorname{Red}(e,S)\) is the maximum semantic similarity between the candidate and already selected evidence.
- \(\lambda\) controls the relevance-diversity trade-off.

The implementation uses \(\lambda=0.6\) and selects three evidence passages for classification.

### 4.8 Retrieval Evaluation

The retrieval component is evaluated using:

| Metric | Description |
|---|---|
| Success@3 | Whether at least one relevant passage occurs in the top three |
| Precision@3 | Fraction of the top three retrieved passages that are relevant |
| Recall@3 | Fraction of relevant passages retrieved in the top three |
| NDCG@3 | Ranking quality with position-sensitive relevance |
| MRR | Reciprocal rank of the first relevant passage |
| MAP | Mean average precision |

The implementation uses available gold evidence passages to construct relevance mappings for retrieval evaluation.

### 4.9 Veracity Classification

The retrieved evidence passages are concatenated and provided together with the claim to a multilingual transformer classifier.

The classification objective is binary:

| Label | Class ID |
|---|---:|
| SUPPORTS | 0 |
| REFUTES | 1 |

Three transformer architectures are evaluated:

| Model | Hugging Face identifier |
|---|---|
| XLM-RoBERTa | `xlm-roberta-base` |
| MuRIL | `google/muril-base-cased` |
| IndicBERTv2 | `ai4bharat/IndicBERTv2-MLM-only` |

Each classifier receives the claim and its retrieved evidence as a paired input.

### 4.10 Class Imbalance Handling and Training

The classification implementation incorporates:

- Inverse-frequency class weighting.
- Focal loss.
- Gradient clipping.
- Learning-rate scheduling.
- Gradient accumulation when necessary.
- Gradient checkpointing.
- Early stopping.
- NaN loss monitoring.
- GPU memory-aware batch-size adjustment.

The focal loss is defined as:

\[
\mathcal{L}_{\mathrm{focal}}
=
-\alpha_t(1-p_t)^\gamma\log(p_t)
\]

The implementation uses \(\gamma=2.0\) and class-dependent weights derived from the training label distribution.

The configured initial maximum sequence length is 256 tokens.

Each model is trained with multiple fallback configurations to handle numerical instability and GPU memory limitations.

### 4.11 Ensemble Learning

The classification models are combined using three ensemble strategies.

**Majority Voting**

The predicted class receiving the highest number of model votes is selected.

**Soft Voting**

The class probabilities are averaged across the available models:

\[
P(y\mid x)=\frac{1}{M}
\sum_{i=1}^{M}P_i(y\mid x)
\]

**Weighted Voting**

The model probabilities are combined using weights derived from the squared development macro-F1 scores:

\[
w_i=
\frac{F1_i^2}{\sum_j F1_j^2}
\]

\[
P(y\mid x)=\sum_i w_iP_i(y\mid x)
\]

The individual models and ensemble approaches are compared using development macro-F1.

The top three approaches are selected for full Subtask 1 submission generation.

---

## 5. Subtask 2: End-to-End Claim Verification

Subtask 2 uses a retrieval-augmented generation pipeline to produce evidence summaries, veracity predictions, and justifications.

### 5.1 Evidence Retrieval

The pipeline uses the selected dense embedding model, FAISS similarity search, BM25, the selected fusion strategy, near-duplicate filtering, and cross-encoder reranking.

The top five evidence passages are retrieved for each claim.

### 5.2 Evidence Summarization

The retrieved passages are combined into a structured context.

The generation model is:

`google/mt5-base`

A summarization prompt instructs the model to summarize the retrieved information using only the supplied evidence and to produce a concise summary.

The intended summary length is approximately 100–150 words.

When the generator cannot be loaded, the implementation uses an extractive fallback that concatenates retrieved evidence and truncates it to a maximum of 150 words.

### 5.3 Veracity Verification

The current implementation uses a heuristic verification function based on:

- Token overlap between the claim and evidence summary.
- The presence of selected negation indicators.

The implementation predicts REFUTES when a configured negation indicator is present or when token overlap is below 0.2; otherwise, it predicts SUPPORTS.

This component is a lightweight heuristic rather than a separately trained natural-language inference model.

### 5.4 Justification Generation

A second generation prompt uses the original claim, evidence summary, and predicted label to produce a natural-language justification.

The intended justification length is approximately 100–120 words.

When the generator is unavailable, an extractive fallback is used.

### 5.5 Structured Output

The generated output contains:

- Evidence summary.
- Prediction.
- Justification.

Each record is associated with the original claim identifier.

The output is serialized into language-specific JSON files.

---

## 6. Models and Technologies

### 6.1 Machine Learning and Deep Learning

| Component | Technology |
|---|---|
| Deep learning framework | PyTorch |
| Transformer models | Hugging Face Transformers |
| Sentence embeddings | Sentence Transformers |
| Dense vector search | FAISS |
| Sparse retrieval | BM25 |
| Lexical representation | TF-IDF |
| Evidence reranking | Cross-Encoder |
| Classification | XLM-RoBERTa, MuRIL, IndicBERTv2 |
| Text generation | mT5 |

### 6.2 Supporting Libraries

- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- WordCloud
- tqdm
- rank-bm25
- Python standard libraries

---

## 7. Experimental Configuration

The following configurations are defined in the implementation.

### 7.1 Classification Models

| Model | Maximum Length | Learning Rate | Initial Batch Size |
|---|---:|---:|---:|
| XLM-RoBERTa-base | 256 | 3e-5 | 16 |
| MuRIL-base-cased | 256 | 2e-5 | 16 |
| IndicBERTv2 | 256 | 2e-5 | 16 |

### 7.2 Retrieval Configuration

| Parameter | Configuration |
|---|---|
| Dense embedding models | 3 candidates |
| Dense retrieval | FAISS IndexFlatIP |
| Sparse retrieval | BM25Okapi |
| Fusion strategies | Weighted fusion and RRF |
| RRF constant | 60 |
| Initial fusion weights | 0.0, 0.25, 0.5, 0.75, 1.0 |
| Near-duplicate threshold | 0.85 |
| MMR lambda | 0.6 |
| Subtask 1 evidence count | 3 |
| Subtask 2 evidence count | 5 |
| Cross-encoder | MS MARCO MiniLM L-12-v2 |

### 7.3 Generation Configuration

| Component | Configuration |
|---|---|
| Generator | google/mt5-base |
| Generation strategy | Deterministic generation |
| Sampling | Disabled |
| Evidence summary target | 100–150 words |
| Justification target | 100–120 words |

### 7.4 Reproducibility

The implementation uses a fixed random seed:

```python
SEED = 42
```

The seed is applied to Python, NumPy, and PyTorch.

The execution device is selected automatically:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

---

## 8. Dataset Organization

The implementation expects the dataset to be organized as follows:

```text
IndicClaimVer/
│
├── SubtaskA/
│   ├── Train/
│   │   ├── train_subtask1.json
│   │   ├── dev_subtask1.json
│   │   └── evidence_pool.json
│   │
│   └── Test/
│       ├── evidence_pool_test.json
│       ├── subtask1_english_test.json
│       ├── subtask1_hindi_test.json
│       ├── subtask1_bengali_test.json
│       └── subtask1_codemix_test.json
│
├── SubtaskB/
│   ├── Train/
│   │   └── dev_subtask2.json
│   │
│   └── Test/
│       ├── subtask2_english_test.json
│       ├── subtask2_hindi_test.json
│       ├── subtask2_bengali_test.json
│       └── subtask2_codemix_test.json
│
└── submissions/
    ├── majority_voting/
    ├── soft_voting/
    ├── weighted_voting/
    └── subtask2_shared/
```

The dataset files are expected to follow the JSON formats supplied by the shared-task organizers.

**Note:** The directory structure above describes the expected input and generated output organization. The datasets are not included in this repository.

---

## 9. Installation

### 9.1 Clone the Repository

```bash
git clone https://github.com/rachanabn20/IndicClaimVer-FIRE2026.git

cd IndicClaimVer-FIRE2026
```

### 9.2 Create a Python Environment

Using Conda:

```bash
conda create -n indicclaimver python=3.10 -y

conda activate indicclaimver
```

### 9.3 Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn wordcloud

pip install scikit-learn

pip install torch

pip install transformers sentence-transformers

pip install faiss-cpu

pip install rank-bm25 tqdm
```

For GPU-based FAISS retrieval, install a compatible GPU-enabled FAISS build where supported by your CUDA and Python environment.

For GPU-enabled PyTorch, install the build appropriate for your CUDA configuration using the official PyTorch installation instructions.

The implementation also supports optional Hugging Face transfer acceleration:

```bash
pip install hf_transfer
```

### 9.4 Hardware Requirements

A CUDA-enabled GPU is recommended for transformer fine-tuning, embedding generation, reranking, and text generation.

CPU execution is supported by the device-selection logic, although large-scale model execution can be considerably slower.

GPU memory availability may affect batch sizes and training configurations.

---

## 10. Execution

### Step 1: Configure Dataset Paths

Open `indicclaimver.ipynb` and update the dataset root:

```python
DATA_ROOT = r"C:/Users/VGST/Desktop/IndicClaimVer"
```

Replace this path with the directory containing the downloaded shared-task datasets.

### Step 2: Initialize the Environment

Execute the initial notebook cells to:

- Import dependencies.
- Configure the computation device.
- Initialize random seeds.
- Verify dataset paths.
- Load the pretrained transformer models.
- Initialize the evidence collections.

### Step 3: Execute Subtask 1 Retrieval

Run the retrieval evaluation and model-selection cells.

The implementation:

1. Loads the evidence pool.
2. Evaluates multilingual embedding models.
3. Constructs the FAISS index.
4. Builds the BM25 index.
5. Tunes weighted fusion.
6. Compares weighted fusion with RRF.
7. Applies near-duplicate filtering.
8. Reranks candidate passages.
9. Selects diverse evidence using MMR.

### Step 4: Train Classification Models

Execute the classification cells to fine-tune:

- XLM-RoBERTa-base.
- MuRIL-base-cased.
- IndicBERTv2.

The notebook evaluates individual classifiers and constructs ensemble predictions.

### Step 5: Generate Subtask 1 Submissions

The implementation selects the top three approaches based on development macro-F1.

For each selected approach, it generates language-specific JSON files containing:

- Claim ID.
- Top three retrieved evidence passages.
- Predicted veracity label.

### Step 6: Generate Subtask 2 Submissions

Execute the RAG pipeline to:

1. Retrieve the top five evidence passages.
2. Generate an evidence summary.
3. Predict the veracity label.
4. Generate a justification.
5. Serialize the results into language-specific JSON files.

### Step 7: Validate Generated Submissions

The notebook includes a submission-validation function that checks:

- Missing identifiers.
- Duplicate identifiers.
- Missing evidence passages.
- Incorrect evidence counts.
- Missing summaries or justifications.
- Invalid prediction labels.
- JSON serialization errors.

Generated files are stored in the configured `submissions` directory.

---

## 11. Output and Submission Files

### 11.1 Subtask 1

The output directory is organized by the selected classification approach.

```text
submissions/
│
├── xlm-roberta-base/
│   ├── English_subtask1.json
│   ├── Hindi_subtask1.json
│   ├── Bengali_subtask1.json
│   └── CodeMix_subtask1.json
│
├── muril/
│   └── ...
│
└── weighted_voting/
    └── ...
```

The actual approach directories depend on the top three configurations selected during development evaluation.

Example record:

```json
{
  "ID": "example_001",
  "top_3_evidence": [
    "First retrieved evidence passage.",
    "Second retrieved evidence passage.",
    "Third retrieved evidence passage."
  ],
  "Prediction": "SUPPORTS"
}
```

### 11.2 Subtask 2

Subtask 2 outputs are stored in a shared directory:

```text
submissions/
└── subtask2_shared/
    ├── English_subtask2.json
    ├── Hindi_subtask2.json
    ├── Bengali_subtask2.json
    └── CodeMix_subtask2.json
```

Example record:

```json
{
  "ID": "example_001",
  "Evidence": "Generated evidence summary.",
  "Prediction": "SUPPORTS",
  "Justification": "Generated justification based on the retrieved evidence."
}
```

The examples above illustrate the implementation's output structure. Actual predictions and evidence are generated by the notebook.

---

## 12. Evaluation Metrics

### 12.1 Evidence Retrieval

The retrieval pipeline uses:

- Success@3
- Precision@3
- Recall@3
- NDCG@3
- Mean Reciprocal Rank (MRR)
- Mean Average Precision (MAP)

### 12.2 Claim Classification

The classification models are evaluated using:

- Accuracy
- Precision
- Recall
- Macro Precision
- Macro Recall
- Macro-F1
- Weighted-F1
- Class-wise F1

The implementation also generates:

- Confusion matrices.
- ROC curves.
- Precision-recall curves.
- Classification reports.

Macro-F1 is used as the primary development metric for comparing classification approaches.

### 12.3 Development-Based Model Selection

The development evaluation is used to select:

- The best dense embedding model.
- The weighted fusion parameter.
- The retrieval fusion strategy.
- The classification model configurations.
- The ensemble approach.

The top three classification approaches are selected for full Subtask 1 submission generation.

---

## 13. Implementation Considerations

The following details are important when reproducing the implementation.

1. **Evidence Pool Selection:** The system prefers the dedicated test evidence pool when available and otherwise falls back to the training evidence collection.

2. **Retrieval Evaluation:** Gold evidence mappings are constructed from the available training and development records. The sampled training claims are also used during retrieval-model and fusion-parameter selection in the provided notebook.

3. **Model Selection:** Classification approaches are ranked using development macro-F1.

4. **GPU Memory Management:** The implementation includes cleanup utilities, CUDA cache clearing, retry mechanisms, and adaptive training configurations.

5. **Generation Fallback:** An extractive fallback is provided when the mT5 generator cannot be loaded.

6. **Submission Schema:** The label terminology and field names must be checked against the official organizer-provided schema before final submission. In particular, the official Subtask 2 description specifies `SUPPORTED`/`REFUTED`, whereas the current implementation's primary prediction logic and validation use `SUPPORTS`/`REFUTES`.

7. **Exception Handling:** The Subtask 2 exception fallback should be aligned with the normal output keys and prediction labels before relying on automated validation.

8. **Reproducibility:** Model downloads, hardware, library versions, and available GPU memory can affect execution time and numerical reproducibility.

---

## 14. Limitations and Future Improvements

Potential directions for further development include:

- Replacing the Subtask 2 token-overlap verification heuristic with a trained multilingual natural-language inference model.
- Incorporating external evidence sources and source provenance information for open-domain verification.
- Improving multilingual lexical retrieval using language-aware tokenization and normalization.
- Exploring fine-tuned multilingual embedding models for evidence retrieval.
- Optimizing evidence selection using learned ranking and reranking objectives.
- Incorporating evidence-level entailment and contradiction detection.
- Improving evidence-grounded generation with explicit factual consistency checks.
- Evaluating retrieval and classification performance separately across languages.
- Conducting ablation studies to quantify the contribution of each retrieval and classification component.

---

## 15. Acknowledgements

We acknowledge the organizers of **IndicClaimVer@FIRE 2026** for providing the shared-task framework, datasets, and evaluation platform.

We also acknowledge the developers and research communities behind PyTorch, Hugging Face Transformers, Sentence Transformers, FAISS, and the other open-source libraries used in this implementation.

---



**Project Links**

- GitHub: https://github.com/rachanabn20/IndicClaimVer-FIRE2026
- Official Shared Task Website: https://sites.google.com/view/indicclaimver2026/home
- FIRE 2026: https://fire.irsi.org.in/fire/2026/home

---

<p align="center">
  <b>IndicClaimVer@FIRE 2026 | Team TokenX</b><br>
  Advancing Multilingual Claim Verification and Evidence Retrieval for Indian Languages.
</p>
