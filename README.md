# Fake News Classification via NLP Models

## Team Members
- **Jean-Denis Drané**
- **Mercy Sneha** 
- **Michael Libio**

## Project Overview
This repository contains a comprehensive exploration of 9 different NLP approaches for binary classification of fake vs real news articles. We systematically developed and compared models ranging from traditional machine learning to state-of-the-art transformer architectures, achieving accuracies from 88.96% to 98.76%.

## Dataset
- **Training Data**: 27,320 news articles (80% split)
- **Validation Data**: 6,831 news articles (20% split)
- **Test Data**: 9,983 news articles 
- **Classes**: Fake (0), Real (1)
- **Distribution**: Well-balanced dataset
- **Text Processing**: Tab-separated format with consistent preprocessing pipeline

## 🏆 Model Results Summary

| Rank | Model | Accuracy | Training Time | Architecture |
|------|-------|----------|---------------|--------------|
| 🥇 1 | **Full BERT Fine-tuned** | **98.76%** | 13.2 min | BERT-base-uncased (109M parameters) |
| 🥈 2 | Simple BERT | 95.87% | 1.2 min | Frozen BERT + LogisticRegression |
| 🥉 3 | **Model 2: TF-IDF + GridSearchCV** | **95.45%** | **0.5 min** | **Pipeline optimization (Best Non-BERT)** |
| 4 | Logistic Regression 4 | 95.01% | <0.1 min | Enhanced hyperparameters |
| 5 | Logistic Regression | 93.99% | <0.1 min | TF-IDF vectorization |
| 6 | Linear SVC | 93.36% | <0.1 min | Support Vector Classifier |
| 7 | Baseline LogisticRegression | 92.90% | <0.1 min | CountVectorizer baseline |
| 8 | Naive Bayes | 92.77% | <0.1 min | Traditional ML |
| 9 | Universal Sentence Encoder + LR | 92.45% | <0.1 min | USE embeddings (512-dim) |
| 10 | Model 6b: GloVe + Pooling + LR | 88.96% | <0.1 min | GloVe-6B-100d embeddings |

## 📊 Performance Categories

### Deep Learning (BERT)
- **Average Accuracy**: 97.32%
- **Best Performance**: Full BERT Fine-tuned at 98.76%
- **Trade-off**: Highest accuracy but longest training time

### Traditional ML 
- **Average Accuracy**: 93.60%
- **Range**: 92.77% - 95.01%
- **Strength**: Fast training (<0.1 min) with strong performance

### Embeddings-based
- **Average Accuracy**: 90.71%
- **Range**: 88.96% - 92.45%
- **Note**: Pre-trained embeddings show variable effectiveness

### Pipeline Experiments
- **Accuracy**: 95.45%
- **Approach**: Systematic hyperparameter optimization

## 🚀 Key Insights & Business Recommendations

### ✅ Production Ready
- **Best Model**: Full BERT achieves 98.76% accuracy - excellent for production
- **Recommended**: Deploy with confidence for high-stakes applications

### ⚡ Fast & Accurate Options (>90% accuracy, <5min training)
For rapid deployment or resource-constrained environments:
- **Model 2: TF-IDF + GridSearchCV**: 95.45% in 0.5 minutes (**Best Non-BERT Model**)
- **NLPModel4 - Logistic Regression**: 95.01% in <0.1 minutes (Best Traditional ML)
- **Simple BERT**: 95.87% in 1.2 minutes
- **Baseline LogisticRegression**: 92.90% in <0.1 minutes

### 📈 Performance Analysis
- **Model Range**: 88.96% - 98.76% (9.80 percentage point spread)
- **Improvement over Baseline**: +5.86 percentage points (92.90% → 98.76%)
- **Efficiency Sweet Spot**: Traditional ML achieves 93-95% accuracy with minimal training time

## 📁 Repository Structure
```
project-nlp-challenge/
├── 📊 Core Notebooks
│   ├── base_model_LR.ipynb                     # Baseline LogisticRegression (92.90%)
│   ├── simple_bert_model.ipynb                 # Simple BERT (95.87%)
│   ├── full_bert_model.ipynb                   # Full BERT Fine-tuned (98.62%)
│   ├── GridSearchCV_pipeline_model_selection.ipynb # Pipeline optimization (95.45%)
│   ├── NLPModel1_LoR.ipynb                     # Logistic Regression (93.99%)
│   ├── NLPModel2_SVM.ipynb                     # Linear SVC (93.36%)
│   ├── NLPModel3_NB.ipynb                      # Naive Bayes (92.77%)
│   ├── NLPModel4_LoR.ipynb                     # Enhanced LR (95.01%)
│   ├── Model6a_USE_LR.ipynb                    # Universal Sentence Encoder (92.45%)
│   ├── Model6b_GloVe_Pooling_LR.ipynb          # GloVe embeddings (88.96%)
│   ├── data_explore.ipynb                      # Data exploration
│   └── results_summary.ipynb                   # Comprehensive analysis
├── 🔧 Utilities & Code
│   ├── preprocess.py                           # Standardized preprocessing
│   └── model_eval.py                           # Model evaluation & visualization
├── 📈 Results & Models
│   ├── results/                                # Model performance results (JSON)
│   │   ├── baseline_lr_results.json
│   │   ├── full_bert_model_results.json
│   │   ├── simple_bert_results.json
│   │   ├── GridSearchCV_pipeline_selection_results.json
│   │   ├── NLPModel1_LoR_results.json
│   │   ├── NLPModel2_SVC_results.json
│   │   ├── NLPModel3_ND_results.json
│   │   ├── NLPModel4_LoR_results.json
│   │   ├── model6a_use_lr_results.json
│   │   └── model6b_glove_pooling_lr_results.json
│   ├── trained_models/                         # Saved traditional ML models
│   ├── prediction/                             # Test predictions
├── 📋 Documentation & Planning
│   ├── README.md                               # This file
│   ├── project_plan.md                         # Project planning
│   └── g2_presentation.pptx                    # Final presentation
└── 🔧 Configuration
    └── .gitignore                              # Git ignore rules
```

### 📥 Setup Instructions for New Users
To reproduce the complete project:

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd project-nlp-challenge
   ```

2. **Download the dataset**:
   ```
   Create data/ folder and add:
   - training_data_lowercase.csv
   - testing_data_lowercase_nolabels.csv
   ```

3. **Run analysis notebooks**:
   ```bash
   jupyter notebook results_summary.ipynb  # Generate charts and analysis
   ```

## 🛠️ Technical Implementation

### Preprocessing Pipeline (`preprocess.py`)
- **Standardized data loading**: Tab-separated parsing with error handling
- **Text cleaning**: Punctuation removal, whitespace normalization
- **Feature extraction**: CountVectorizer, TF-IDF, and embedding support
- **Consistent splits**: Stratified train/validation maintaining label balance
- **Reusable functions**: Used across all 9 model implementations

### Model Evaluation System (`model_eval.py`)
- **Standardized metrics**: Accuracy, training time, confusion matrices
- **Visualization tools**: Automated chart generation for comparisons
- **Results storage**: JSON format for easy analysis and reproduction

### Hardware & Training Environments
- **Apple Silicon (MPS)**: BERT models and baseline model training
  - Full BERT Fine-tuned, Simple BERT, Baseline LogisticRegression
  - M4 Pro with MPS acceleration for efficient deep learning
- **Windows/Google Colab**: Traditional ML and embedding-based models
  - TF-IDF models, SVM, Naive Bayes, embeddings experiments
  - Distributed development across team members and platforms

## 🔬 Model Deep Dives

### 🥇 Champion: Full BERT Fine-tuned (98.76%)
```python
# Key hyperparameters for best model (trained on Apple Silicon MPS)
hyperparameters = {
    'bert_model': 'bert-base-uncased',
    'max_length': 39,
    'num_train_epochs': 3,
    'per_device_train_batch_size': 16,
    'learning_rate': 5e-05,
    'weight_decay': 0.01,
    'warmup_steps': 500,
    'early_stopping_patience': 2
}
```
- **Performance**: 98.76% accuracy (99.91% training accuracy)
- **Architecture**: 109M parameters, fine-tuned end-to-end
- **Training**: 13.2 minutes on Apple Silicon MPS
- **Device**: MPS acceleration with 30,522 vocab size

### ⚡ Efficiency Leader: GridSearchCV Pipeline (95.45% in 0.5min)
```python
# Best pipeline configuration found through systematic search
best_params = {
    'clf': 'LogisticRegression(max_iter=1000)',
    'clf__C': 10,
    'tfidf__max_df': 0.7,
    'tfidf__min_df': 1,
    'tfidf__ngram_range': [1, 2]
}
```
- **Strategy**: Systematic hyperparameter optimization across multiple algorithms
- **Performance**: 95.45% accuracy (99.96% training accuracy) - **Best non-BERT model**
- **Efficiency**: 0.5 minute training vs 13.2 minutes for BERT (**26x faster**)
- **Platform**: Windows/Google Colab training
- **Components**: TF-IDF + GridSearchCV across LR, SVC, NB
- **Achievement**: Demonstrates the power of systematic hyperparameter search
- **Ranking**: 3rd overall, **best traditional ML approach**

### 🏆 Best Single Algorithm: NLPModel4 - Tuned Logistic Regression (95.01%)
```python
# Best individual model with optimized hyperparameters
hyperparameters = {
    'token_pattern': '(?u)\\b\\w+\\b',
    'ngram_range': [1, 2],        # Unigrams and bigrams
    'min_df': 2,                  # Remove rare terms
    'max_df': 0.8,                # Remove too common terms  
    'max_features': 10000         # Feature limit
}
```
- **Key Innovation**: Manual hyperparameter tuning of TF-IDF + LogisticRegression
- **Performance**: 95.01% accuracy - very close to GridSearchCV performance
- **Efficiency**: <0.1 minute training time (**132x faster than BERT**)
- **Sweet Spot**: Best individual algorithm for ultra-fast deployment
- **Training**: Windows/Google Colab platforms
- **Ranking**: 4th overall, best single traditional ML model

### 💡 Baseline Excellence: CountVectorizer + LogisticRegression (92.90%)
- **Architecture**: Simple CountVectorizer (10k features) + LogisticRegression
- **Training**: Apple Silicon MPS, <0.1 minute
- **Insight**: Proper preprocessing enables competitive traditional ML
- **Use case**: Rapid prototyping, resource-constrained deployments
- **Foundation**: Demonstrates strong baseline for comparison

## 📚 Usage Examples

### Quick Start - Best Traditional ML Model
```python
from preprocess import load_and_parse_data, create_train_validation_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load and preprocess data
train_data = load_and_parse_data('data/training_data_lowercase.csv')
X_train, X_val, y_train, y_val = create_train_validation_split(train_data)

# NLPModel4: TF-IDF + Optimized Logistic Regression pipeline (achieves 95.01%)
vectorizer = TfidfVectorizer(
    token_pattern=r'(?u)\b\w+\b',
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.8,
    max_features=10000
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_tfidf, y_train)
accuracy = model.score(X_val_tfidf, y_val)  # ~95.01%
```

### Load and Analyze Results
```python
from model_eval import load_all_model_results, plot_best_models_comparison

# Load all saved model results (available in results/ folder)
models = load_all_model_results()
print(f"Loaded {len(models)} model results")

# Generate comparison charts
plot_best_models_comparison(models, save_path="model_comparison.png")
```

## 🎯 Monitoring & Deployment Recommendations

### Production Deployment
- **Primary**: Full BERT Fine-tuned for maximum accuracy (98.76%)
- **High-efficiency alternative**: NLPModel4 for 95%+ accuracy with minimal resources
- **Monitoring**: Set alerts below 90% accuracy threshold

### Operational Considerations
- **Confidence scoring**: Implement prediction confidence tracking
- **Human-in-the-loop**: Route low-confidence predictions for review
- **Retraining schedule**: Monthly/quarterly model updates
- **A/B testing**: Compare against existing solutions

### Cost-Benefit Analysis
- **BERT**: 98.76% accuracy, 13.2 min training - for maximum performance
- **GridSearchCV**: 95.45% accuracy, 0.5 min training - **best efficiency/accuracy balance**
- **NLPModel4**: 95.01% accuracy, <0.1 min training - for ultra-fast deployment
- **Key Trade-off**: 3.31 percentage points vs 26x faster training (GridSearchCV vs BERT)

## 🔧 Technologies Used
- **Deep Learning**: PyTorch, Transformers (Hugging Face) 
- **Traditional ML**: scikit-learn, pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Hardware**: 
  - Apple Silicon MPS acceleration (BERT models, baseline)
  - Windows/Google Colab (traditional ML, embeddings)
- **Development**: Distributed team collaboration across platforms
- **Version Control**: Git with comprehensive model tracking

## 📈 Future Improvements
1. **Ensemble methods**: Combine top-performing models (BERT + NLPModel4)
2. **Data augmentation**: Expand training dataset with synthetic examples
3. **Advanced architectures**: Explore RoBERTa, DistilBERT for efficiency
4. **Deployment optimization**: Model compression and quantization for BERT
5. **Real-time inference**: API development for production use
6. **Cross-validation**: Implement k-fold validation for more robust evaluation

---

*This project demonstrates the complete machine learning lifecycle for fake news classification, from data exploration through model comparison to production recommendations. Our systematic approach identified that while BERT achieves the highest accuracy (98.76%), traditional ML models can achieve 95%+ accuracy with significantly faster training, providing excellent alternatives for different deployment scenarios. The key finding is that proper hyperparameter tuning (NLPModel4) can bridge most of the gap between traditional ML and deep learning while maintaining computational efficiency.*
