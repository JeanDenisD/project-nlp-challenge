# Fake News Classification via NLP Models

## Team Members
- **Jean-Denis Drané**
- **Mercy Sneha**
- **Michael Libio**

## Project Overview
This repository contains exploration of different NLP approaches for binary classification of fake vs real news articles. We systematically developed and compared multiple models ranging from traditional machine learning to state-of-the-art transformer architectures.

## Dataset
- **Training Data**: 34,151 news articles (lowercase preprocessed)
- **Test Data**: 9,983 news articles 
- **Classes**: Fake (0), Real (1)
- **Distribution**: 51.5% fake, 48.5% real (well-balanced)
- **Text Length**: Average 11.7 words per article

## Model Approaches Explored

### 1. Baseline Logistic Regression (`base_model_LR.ipynb`)
- Traditional ML with stemming and CountVectorizer
- **Accuracy**: ~70-75% (baseline)
- NLTK preprocessing with PorterStemmer

### 2. Simple BERT Feature Extraction (`simple_bert_model.ipynb`)
- BERT as frozen feature extractor + LogisticRegression
- **Accuracy**: 95.87%
- Fast training (1.18 minutes), minimal complexity

### 3. Full BERT Fine-tuning (`full_bert_model.ipynb`) 
- Complete BERT-base-uncased fine-tuning
- **Accuracy**: 98.59% (best performance)
- Training time: 11.29 minutes on M4 Pro

## Key Technical Components

### Preprocessing Pipeline (`preprocess.py`)
- Tab-separated data parsing 
- Text cleaning and tokenization
- Train/validation splits with stratification
- TF-IDF vectorization for traditional ML
- Reusable functions for both traditional ML and BERT approaches

### Training Optimizations
- **Full BERT**: Hugging Face Trainer with early stopping
- **Simple BERT**: Frozen embeddings for efficiency
- **Hardware**: MPS acceleration on Apple Silicon

## Results Summary

| Model | Architecture | Training Time | Accuracy | Key Innovation |
|-------|-------------|---------------|----------|----------------|
| Baseline | TF-IDF + LogisticRegression | ~minutes | ~70-75% | Traditional ML |
| Simple BERT | Frozen BERT + Classifier | 1.18 min | 95.87% | Feature extraction |
| **Full BERT** | **Fine-tuned BERT-base** | **11.29 min** | **98.59%** | **Complete fine-tuning (best)** |

## Repository Structure
```
├── preprocess.py                        # Main preprocessing utilities
├── base_model_LR.ipynb                  # Baseline Logistic Regression
├── simple_bert_model.ipynb              # Simple BERT feature extraction
├── full_bert_model.ipynb                # Full BERT fine-tuning
├── data/
│   ├── training_data_lowercase.csv      # Training dataset
│   └── testing_data_lowercase_nolabels.csv # Test dataset
└── README.md                            # Project documentation
```

## Key Insights
1. **BERT power**: Even frozen BERT features achieve 95.87% accuracy
2. **Efficiency tradeoff**: Simple BERT is 9.5x faster with only 2.7% accuracy loss
3. **Fine-tuning impact**: Full BERT achieves near-perfect 98.59% accuracy
4. **Hardware advantage**: M4 Pro with MPS acceleration enables fast BERT training
5. **Text preprocessing**: Minimal cleaning works best for BERT models

## Performance Comparison
- **Baseline → Simple BERT**: +25.87 percentage points improvement
- **Simple BERT → Full BERT**: +2.73 percentage points improvement  
- **Simple BERT efficiency**: 9.5x faster training than Full BERT
- **Full BERT confidence**: 99% precision/recall for both classes

## Usage
```python
from preprocess import load_and_parse_data, complete_preprocessing_pipeline

# Load and preprocess data
train_data = load_and_parse_data('data/training_data_lowercase.csv')

# Traditional ML pipeline
processed_data = complete_preprocessing_pipeline(
    'data/training_data_lowercase.csv',
    'data/testing_data_lowercase_nolabels.csv'
)

# For BERT models, use the same preprocessing functions
# with minimal text cleaning inside the notebook
```

## Technologies Used
- **PyTorch**: Deep learning framework for BERT models
- **Transformers**: Hugging Face library for BERT
- **scikit-learn**: Traditional ML and evaluation metrics
- **pandas/numpy**: Data processing and manipulation

---
*This project demonstrates the power of transformer models for text classification, achieving near-perfect accuracy on fake news detection through systematic model comparison and optimization.*
