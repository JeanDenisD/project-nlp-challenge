# Fake News Classification Project Plan

## Project Overview
- **Task**: Binary classification of fake (0) vs real (1) news articles  
- **Training Data**: `training_data_lowercase.csv` (34,151 articles)
- **Test Data**: `testing_data_lowercase_nolabels.csv` (9,983 articles) 
- **Team**: 3 people, 3 days total

---

## Day 1: Setup & Baseline (All Together)

### Data & Preprocessing
- [ ] Load and explore both datasets
- [ ] Basic text cleaning and preprocessing (based on model - it may need adjustment)
- [ ] Create train/validation split
- [ ] Understand data structure

### Baseline Model
- [ ] Build simple TF-IDF + Logistic Regression
- [ ] Evaluate performance (accuracy, F1-score)
- [ ] Document baseline results

---

## Day 2: Individual Models

### Model Options (Pick 1-2)
- TF-IDF + Random Forest/SVM
- Basic Neural Network + embeddings
- Simple BERT/transformer
- Count Vectorizer + Naive Bayes

### Standard Evaluation
- [ ] Use same train/validation split
- [ ] Report: accuracy, precision, recall, F1-score
- [ ] Document results consistently

### Individual Work
**JD:**
- [ ] Implement chosen model(s) and evaluate

**Mercy:**
- [ ] Implement chosen model(s) and evaluate  

**Michael:**
- [ ] Implement chosen model(s) and evaluate

---

## Day 3: Final Results

### Compare & Select
- [ ] Compare all model results
- [ ] Choose best performing model
- [ ] Simple ensemble if helpful

### Submit Predictions
- [ ] Generate predictions for test data
- [ ] Create final submission file
- [ ] Estimate accuracy for teacher

### Presentation
- [ ] 10-minute team presentation
- [ ] Show approach, results, insights

---

## Deliverables
- [ ] Python code (documented)
- [ ] Predictions file
- [ ] Accuracy estimation
- [ ] Team presentation
