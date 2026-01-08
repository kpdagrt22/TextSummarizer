# End-to-End NLP Text Summarization with Hugging Face

A production-ready text summarization pipeline built with Hugging Face Transformers. This project demonstrates fine-tuning state-of-the-art models on conversational data and deploying them as a robust, scalable service.

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Transformers](https://img.shields.io/badge/🤗-transformers-orange.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🎯 Overview

This project provides an end-to-end workflow for building a custom text summarization model, from data ingestion to deployment. It uses pretrained transformer models (Pegasus, T5, FLAN-T5) and fine-tunes them on the SAMSum conversational dataset to generate high-quality abstractive summaries.

### Key Features

- **State-of-the-art Models**: Leverage pretrained transformers from Hugging Face Model Hub
- **Conversational Focus**: Fine-tuned on SAMSum dataset for dialogue and chat summarization
- **Modular Architecture**: Clean, reusable components for data processing, training, and inference
- **Flexible Backbone**: Easy switching between Pegasus, T5, FLAN-T5, and other models
- **Production Ready**: Includes REST API, batch processing, and containerization support
- **Comprehensive Evaluation**: ROUGE metrics and custom evaluation pipelines

## 📋 Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Dataset](#dataset)
- [Models](#models)
- [Usage](#usage)
- [Configuration](#configuration)
- [Development Workflow](#development-workflow)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- 8GB+ RAM

### Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/text-summarizer.git
cd text-summarizer

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Login to Hugging Face for model access
huggingface-cli login
```

## 📁 Project Structure

```
text-summarizer/
├── README.md
├── requirements.txt
├── setup.py
├── .env.example                 # Environment variables template
├── Dockerfile                   # Container configuration
├── config/
│   ├── config.yaml             # Main configuration file
│   ├── params.yaml             # Training hyperparameters
│   └── logging.yaml            # Logging configuration
├── data/
│   ├── raw/                    # Original SAMSum dataset
│   ├── processed/              # Tokenized and preprocessed data
│   └── samples/                # Test samples for quick validation
├── notebooks/
│   ├── 01_eda_samsum.ipynb    # Exploratory data analysis
│   ├── 02_model_comparison.ipynb  # Model performance comparison
│   └── 03_error_analysis.ipynb    # Failed predictions analysis
├── src/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── configuration.py   # Configuration manager
│   │   └── entity.py          # Config entity classes
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   └── model_evaluation.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── training_pipeline.py
│   │   └── prediction_pipeline.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── common.py          # Utility functions
│   │   └── logger.py          # Custom logging
│   └── api.py                  # FastAPI application
├── scripts/
│   ├── run_pipeline.sh         # Full pipeline execution
│   ├── train.sh                # Training script
│   ├── evaluate.sh             # Evaluation script
│   └── serve.sh                # API server launcher
├── models/
│   ├── checkpoints/            # Fine-tuned model weights
│   └── tokenizer/              # Saved tokenizer configs
├── tests/
│   ├── unit/
│   │   ├── test_data.py
│   │   ├── test_models.py
│   │   └── test_pipeline.py
│   └── integration/
│       └── test_api.py
├── examples/
│   ├── sample_conversation.txt
│   ├── sample_summary.txt
│   └── api_examples.py
└── artifacts/                   # Training artifacts and logs
    ├── logs/
    └── metrics/
```

## ⚡ Quick Start

### 1. Run the Complete Pipeline

```bash
# Execute the full workflow (data → training → evaluation)
python main.py
```

### 2. Use Pre-trained Model for Inference

```python
from src.pipeline.prediction_pipeline import PredictionPipeline

# Initialize pipeline
predictor = PredictionPipeline(model_path="models/checkpoints/best-model")

# Summarize text
conversation = """
John: Hey, are we still meeting at 3pm today?
Sarah: Yes! I'll bring the project files.
John: Perfect. Should we invite Mike too?
Sarah: Good idea, I'll send him a message now.
"""

summary = predictor.predict(conversation)
print(f"Summary: {summary}")
```

### 3. Start the REST API

```bash
# Launch FastAPI server
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

# Test the endpoint
curl -X POST "http://localhost:8000/summarize" \
  -H "Content-Type: application/json" \
  -d '{"text": "Your long text here..."}'
```

## 📊 Dataset

### SAMSum Corpus

- **Source**: [Hugging Face Datasets](https://huggingface.co/datasets/knkarthick/samsum)
- **Description**: Messenger-style conversations with human-written summaries
- **Size**: ~16,000 conversation-summary pairs
- **Task**: Abstractive dialogue summarization
- **Split**: Train (14,732) / Validation (818) / Test (819)

**Sample Entry**:
```json
{
  "dialogue": "Amanda: I baked cookies. Do you want some?\nJerry: Sure!\nAmanda: I'll bring you tomorrow :-)",
  "summary": "Amanda baked cookies and will bring Jerry some tomorrow."
}
```

### Using Custom Datasets

To use your own dataset, modify `src/components/data_ingestion.py`:

```python
# Load custom CSV/JSON
custom_data = load_custom_dataset("path/to/your/data.csv")
```

## 🤖 Models

### Supported Architectures

| Model | Size | Best For | Training Time* |
|-------|------|----------|----------------|
| **google/pegasus-cnn_dailymail** | 568M | News articles, formal text | ~4 hours |
| **t5-small** | 60M | Quick experiments, low resources | ~1 hour |
| **facebook/bart-large-cnn** | 406M | Balanced performance | ~3 hours |
| **Falconsai/text_summarization** | 60M | Dialogue, conversational text | ~1 hour |

*Approximate training time on SAMSum with single V100 GPU

### Model Selection

```python
# In config/config.yaml
model:
  name: "google/pegasus-cnn_dailymail"
  max_input_length: 512
  max_target_length: 128
```

## 💻 Usage

### Training a Model

```bash
# Using command line
python -m src.pipeline.training_pipeline \
  --model_name google/pegasus-cnn_dailymail \
  --dataset knkarthick/samsum \
  --epochs 3 \
  --batch_size 4 \
  --output_dir models/checkpoints/pegasus-samsum

# Or use the shell script
bash scripts/train.sh
```

### Evaluation

```bash
# Evaluate on test set
python -m src.components.model_evaluation \
  --model_path models/checkpoints/pegasus-samsum \
  --dataset knkarthick/samsum

# Output: ROUGE-1, ROUGE-2, ROUGE-L scores
```

### Batch Prediction

```python
from src.pipeline.prediction_pipeline import BatchPrediction

# Process multiple texts
texts = [
    "First conversation...",
    "Second conversation...",
    "Third conversation..."
]

batch_predictor = BatchPrediction("models/checkpoints/best-model")
summaries = batch_predictor.predict_batch(texts, batch_size=8)
```

## ⚙️ Configuration

### config.yaml

```yaml
data:
  dataset_name: knkarthick/samsum
  train_batch_size: 4
  eval_batch_size: 8
  max_source_length: 512
  max_target_length: 128

model:
  name: google/pegasus-cnn_dailymail
  
training:
  num_epochs: 3
  learning_rate: 2e-5
  warmup_steps: 500
  weight_decay: 0.01
  gradient_accumulation_steps: 4
  
paths:
  data_dir: data/
  model_dir: models/checkpoints/
  log_dir: artifacts/logs/
```

### params.yaml

```yaml
TrainingArguments:
  num_train_epochs: 3
  per_device_train_batch_size: 4
  per_device_eval_batch_size: 8
  warmup_steps: 500
  learning_rate: 2e-5
  evaluation_strategy: "epoch"
  save_strategy: "epoch"
  load_best_model_at_end: true
  metric_for_best_model: "rouge1"
```

## 🔄 Development Workflow

### Standard Pipeline

1. **Configuration Setup**: Define parameters in `config.yaml` and `params.yaml`
2. **Entity Definition**: Create configuration entities in `src/config/entity.py`
3. **Configuration Manager**: Implement in `src/config/configuration.py`
4. **Component Development**:
   - Data Ingestion (`src/components/data_ingestion.py`)
   - Data Transformation (`src/components/data_transformation.py`)
   - Model Training (`src/components/model_trainer.py`)
   - Model Evaluation (`src/components/model_evaluation.py`)
5. **Pipeline Creation**:
   - Training Pipeline (`src/pipeline/training_pipeline.py`)
   - Prediction Pipeline (`src/pipeline/prediction_pipeline.py`)
6. **API Development**: REST endpoints in `src/api.py`
7. **Testing**: Unit and integration tests
8. **Deployment**: Containerization and cloud deployment

## 🌐 API Reference

### Endpoints

#### POST /summarize
Generate summary for input text.

**Request**:
```json
{
  "text": "Your long conversation or article here...",
  "max_length": 128,
  "min_length": 30
}
```

**Response**:
```json
{
  "summary": "Concise summary of the input text.",
  "model": "pegasus-samsum",
  "processing_time": 0.45
}
```

#### POST /batch-summarize
Process multiple texts at once.

#### GET /health
Check API health status.

#### GET /models
List available models.

## 📈 Performance Metrics

### Baseline Results on SAMSum Test Set

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|-------|---------|---------|---------|
| Pegasus-CNN/DM (fine-tuned) | 42.5 | 20.8 | 34.2 |
| T5-small (fine-tuned) | 40.1 | 18.5 | 32.7 |
| BART-large-CNN (fine-tuned) | 43.2 | 21.4 | 35.1 |

## 🐳 Docker Deployment

```bash
# Build image
docker build -t text-summarizer:latest .

# Run container
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  text-summarizer:latest

# Access API
curl http://localhost:8000/health
```

## 🗺️ Roadmap

- [x] Core summarization pipeline
- [x] SAMSum fine-tuning
- [x] REST API implementation
- [ ] Support for CNN/DailyMail and XSum datasets
- [ ] Multi-model ensemble predictions
- [ ] Hugging Face Hub integration for model versioning
- [ ] Streamlit/Gradio web interface
- [ ] Kubernetes deployment configurations
- [ ] Real-time streaming summarization
- [ ] Multi-language support

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure:
- Code follows PEP 8 style guidelines
- All tests pass (`pytest tests/`)
- Documentation is updated
- Commit messages are descriptive

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Third-Party Licenses

- Hugging Face Transformers: Apache 2.0
- SAMSum Dataset: Non-commercial research purposes
- PEGASUS Model: Apache 2.0

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for the Transformers library
- SAMSum dataset creators
- The open-source NLP community

## 📧 Contact

- **Project Maintainer**: Your Name
- **Email**: your.email@example.com
- **GitHub**: [@your-username](https://github.com/your-username)
- **Issues**: [GitHub Issues](https://github.com/your-username/text-summarizer/issues)

---

⭐ If you find this project helpful, please consider giving it a star!
