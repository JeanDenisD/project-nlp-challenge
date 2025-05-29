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
- Traditional ML with standardized preprocessing and CountVectorizer
- **Accuracy**: 92.90% (strong baseline)
- Uses standardized `preprocess.py` functions for consistency
- Enhanced from original NLTK stemming approach

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
- Standardized data loading with tab-separated parsing
- Consistent text cleaning (punctuation removal, whitespace normalization)
- Stratified train/validation splits maintaining label balance
- CountVectorizer and TF-IDF vectorization options
- Reusable functions across all model approaches for consistency

### Training Optimizations
- **Full BERT**: Hugging Face Trainer with early stopping
- **Simple BERT**: Frozen embeddings for efficiency
- **Hardware**: MPS acceleration on Apple Silicon

## Results Summary

| Model | Architecture | Training Time | Accuracy | Key Innovation |
|-------|-------------|---------------|----------|----------------|
| **Baseline** | **CountVectorizer + LogisticRegression** | **~1-2 min** | **92.90%** | **Standardized preprocessing** |
| Simple BERT | Frozen BERT + Classifier | 1.18 min | 95.87% | Feature extraction |
| **Full BERT** | **Fine-tuned BERT-base** | **11.29 min** | **98.59%** | **Complete fine-tuning (best)** |

## Repository Structure
```
├── preprocess.py                        # Standardized preprocessing utilities
├── base_model_LR.ipynb                  # Refactored Logistic Regression (92.90%)
├── base_model_LR_original.ipynb         # Original baseline (backup)
├── simple_bert_model.ipynb              # Simple BERT feature extraction (95.87%)
├── full_bert_model.ipynb                # Full BERT fine-tuning (98.59%)
├── data/
│   ├── training_data_lowercase.csv      # Training dataset (34,151 articles)
│   └── testing_data_lowercase_nolabels.csv # Test dataset (9,983 articles)
└── README.md                            # Project documentation
```

## Key Insights
1. **Strong traditional ML baseline**: Proper preprocessing achieves 92.90% accuracy with simple LogisticRegression
2. **BERT incremental gains**: BERT provides substantial but diminishing returns over strong baselines
3. **Preprocessing importance**: Standardized text cleaning significantly improves traditional ML performance
4. **Efficiency considerations**: Baseline model trains in ~1-2 minutes vs 11+ minutes for full BERT
5. **Architecture evolution**: CountVectorizer → Frozen BERT → Fine-tuned BERT shows clear progression
6. **Hardware advantage**: M4 Pro with MPS acceleration enables efficient BERT experimentation

## Performance Comparison
- **Traditional ML Excellence**: 92.90% accuracy proves traditional methods remain highly competitive
- **Baseline → Simple BERT**: +2.97 percentage points improvement (92.90% → 95.87%)
- **Simple BERT → Full BERT**: +2.72 percentage points improvement (95.87% → 98.59%)
- **Training efficiency**: Baseline is 5-10x faster than BERT approaches
- **Diminishing returns**: Each model upgrade provides smaller improvements
- **Cost-benefit**: Baseline achieves 94% of full BERT performance at 10% of training time

## Usage
```python
from preprocess import load_and_parse_data, create_train_validation_split, clean_text

# Load and preprocess data using standardized functions
train_data = load_and_parse_data('data/training_data_lowercase.csv')

# Apply standardized text cleaning
for item in train_data:
    item['clean_text'] = clean_text(item['text'])

# Create consistent train/validation splits
data_for_split = [{'label': item['label'], 'text': item['clean_text']} for item in train_data]
X_train, X_val, y_train, y_val = create_train_validation_split(data_for_split)

# Traditional ML pipeline with CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

vectorizer = CountVectorizer(max_features=10000, stop_words='english', 
                           lowercase=False, min_df=2, max_df=0.95)
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_vec, y_train)
# Achieves 92.90% validation accuracy
```

## Technologies Used
- **PyTorch**: Deep learning framework for BERT models
- **Transformers**: Hugging Face library for BERT
- **scikit-learn**: Traditional ML and evaluation metrics
- **pandas/numpy**: Data processing and manipulation

---
*This project demonstrates a comprehensive comparison of NLP approaches for fake news classification, from highly optimized traditional ML (92.90% accuracy) to state-of-the-art transformers (98.59% accuracy), emphasizing the importance of proper preprocessing and the cost-benefit analysis of different architectural choices.*
