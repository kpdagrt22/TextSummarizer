# End-to-End NLP Text Summarization Pipeline

> **Learning project:** Modular NLP text summarization pipeline built by [Prakash Kantumutchu (KP)](https://thisiskp.in) while practicing end-to-end ML project structuring with Hugging Face Transformers and FastAPI.  
> **Course / tutorial credit:** Based on [Krish Naik's](https://github.com/krishnaik06) End-to-End Text Summarization MLOps tutorial workflow using `google/pegasus-cnn_dailymail` and the SAMSum conversational dataset.

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Transformers](https://img.shields.io/badge/🤗-transformers-orange.svg)](https://huggingface.co/transformers/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

---

## Overview

This repository structures a conversational text summarization workflow into modular Python stages — from notebook experimentation (`research/`) to typed configuration entities (`src/textSummarizer/entity/`) and a FastAPI inference wrapper (`app.py`):

1. **Stage 1 — Data Ingestion (`stage_1_data_ingestion_pipeline.py`):** Downloads and extracts the SAMSum dialogue dataset archive defined in `config/config.yaml`.
2. **Stage 2 — Data Transformation (`stage_2_data_transformation_pipeline.py`):** Tokenizes dialogues and reference summaries using `AutoTokenizer` (`google/pegasus-cnn_dailymail`).
3. **Stage 3 — Model Trainer (`stage_3_model_trainer_pipeline.py`):** Fine-tunes `AutoModelForSeq2SeqLM` using Hugging Face `Trainer` and hyperparameters from `params.yaml`.
4. **Stage 4 — Model Evaluation (`stage_4_model_evaluation.py`):** Computes ROUGE metrics (`rouge1`, `rouge2`, `rougeL`, `rougeLsum`) on the test split and writes them to CSV.
5. **FastAPI Application (`app.py`):** Exposes `/train` and `/predict` endpoints around `PredictionPipeline`.

---

## Project Structure

```text
├── app.py                                  # FastAPI server (/train and /predict endpoints)
├── main.py                                 # Sequential runner for Stages 1–4
├── config/
│   └── config.yaml                         # Artifact paths, dataset URL, and tokenizer/model names
├── params.yaml                             # TrainingArguments hyperparameters
├── research/                               # Step-by-step Jupyter notebooks prior to modularization
│   ├── 1_data_ingestion.ipynb
│   ├── 2_data_transformation.ipynb
│   ├── 3_model_trainer.ipynb
│   ├── 4_model_evaluation.ipynb
│   ├── research.ipynb
│   └── textsummarizer.ipynb
├── src/textSummarizer/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   └── model_evaluation.py
│   ├── config/
│   │   └── configuration.py
│   ├── constants/
│   ├── entity/
│   ├── logging/
│   ├── pipeline/
│   │   ├── stage_1_data_ingestion_pipeline.py
│   │   ├── stage_2_data_transformation_pipeline.py
│   │   ├── stage_3_model_trainer_pipeline.py
│   │   ├── stage_4_model_evaluation.py
│   │   └── predicition_pipeline.py
│   └── utils/
│       └── common.py
├── Dockerfile
├── requirements.txt
├── setup.py
└── LICENSE                                 # GPL-3.0
```

---

## Local Setup & Usage

```bash
# 1. Create a virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Run the 4-stage pipeline (Data Ingestion → Transformation → Training → Evaluation)
python main.py

# 3. Start the FastAPI server (Swagger UI at http://localhost:8080/docs)
python app.py
```

---

## Credits & License

- **Tutorial credit:** [Krish Naik — End-to-End Text Summarization Project](https://github.com/krishnaik06).
- **Pretrained model:** [`google/pegasus-cnn_dailymail`](https://huggingface.co/google/pegasus-cnn_dailymail) (Apache-2.0).
- **Dataset:** [SAMSum Corpus](https://huggingface.co/datasets/samsum).
- **Repository license:** GNU General Public License v3.0 — see [`LICENSE`](LICENSE).
