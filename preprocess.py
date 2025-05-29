"""
Text Preprocessing Pipeline for Fake News Classification
Based on the 3-Day Fake News Classification Project Plan

This module contains all preprocessing functions needed to prepare
training and test data for fake news classification models.
"""

import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from collections import Counter


def load_and_parse_data(csv_file):
    """
    Load and parse CSV data with tab-separated format.
    
    Expected formats:
    - Training data: "label\ttext" (0 for fake, 1 for real)
    - Test data: "index\ttext" (no labels)
    
    Args:
        csv_file (str): Path to CSV file
        
    Returns:
        list: List of dictionaries with 'label' and 'text' keys
    """
    print(f"Loading data from {csv_file}...")
    
    # Load CSV file
    df = pd.read_csv(csv_file)
    
    # Parse tab-separated format
    parsed_data = []
    
    for index, row in df.iterrows():
        # Get the first column content
        text_content = str(row.iloc[0])
        
        # Parse based on format
        if text_content.startswith('0\t') or text_content.startswith('1\t'):
            # Training data format: "label\ttext"
            label = int(text_content[0])
            content = text_content[2:]  # Skip "label\t"
        elif '\t' in text_content and text_content[0].isdigit():
            # Test data format: "index\ttext"
            tab_index = text_content.find('\t')
            content = text_content[tab_index + 1:]
            label = None  # No label for test data
        else:
            # Fallback: use entire content as text
            content = text_content
            label = None
        
        parsed_data.append({
            'label': label,
            'text': content
        })
    
    print(f"Loaded {len(parsed_data)} articles")
    return parsed_data


def clean_text(text):
    """
    Basic text cleaning function.
    
    Args:
        text (str): Raw text to clean
        
    Returns:
        str: Cleaned text
    """
    # Handle missing values
    if pd.isna(text) or text == '' or text is None:
        return ""
    
    # Convert to string (data already lowercase - skip conversion)
    text = str(text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove special characters (keep only letters, numbers, spaces)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    return text


# Removed advanced_preprocess and extract_features as they're not used in the base model


def create_train_validation_split(data, test_size=0.2, random_state=42):
    """
    Create train/validation split while maintaining label balance.
    
    Args:
        data (list): List of data dictionaries with 'label' and 'text'
        test_size (float): Proportion of data for validation
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: X_train, X_val, y_train, y_val
    """
    # Extract texts and labels
    texts = [item['text'] for item in data]
    labels = [item['label'] for item in data]
    
    # Create stratified split to maintain label balance
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=labels
    )
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Train label distribution: {Counter(y_train)}")
    print(f"Validation label distribution: {Counter(y_val)}")
    
    return X_train, X_val, y_train, y_val


def create_tfidf_pipeline(X_train, max_features=10000):
    """
    Create and fit TF-IDF vectorizer on training data.
    
    Args:
        X_train (list): Training text data
        max_features (int): Maximum number of features to extract
        
    Returns:
        tuple: (vectorizer, X_train_vectorized)
    """
    print("Creating TF-IDF vectorizer...")
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        lowercase=False,          # Data already lowercase
        stop_words='english',     # Remove common words
        ngram_range=(1, 2),       # Use unigrams and bigrams
        min_df=2,                 # Ignore words appearing in < 2 documents
        max_df=0.95,              # Ignore words appearing in > 95% documents
        strip_accents='unicode',  # Handle special characters
        token_pattern=r'\b\w+\b'  # Word boundaries
    )
    
    # Fit on training data
    X_train_vectorized = vectorizer.fit_transform(X_train)
    
    print(f"TF-IDF feature matrix shape: {X_train_vectorized.shape}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    return vectorizer, X_train_vectorized


def complete_preprocessing_pipeline(train_file, test_file=None, max_features=10000):
    """
    Complete preprocessing pipeline for fake news classification.
    
    Args:
        train_file (str): Path to training data CSV
        test_file (str, optional): Path to test data CSV
        max_features (int): Maximum TF-IDF features
        
    Returns:
        dict: Dictionary containing all preprocessed data and vectorizer
    """
    print("=== Starting Complete Preprocessing Pipeline ===")
    
    # 1. Load and parse data
    train_data = load_and_parse_data(train_file)
    test_data = load_and_parse_data(test_file) if test_file else None
    
    # 2. Clean text data
    print("\nCleaning training data...")
    for item in train_data:
        item['text'] = clean_text(item['text'])
    
    if test_data:
        print("Cleaning test data...")
        for item in test_data:
            item['text'] = clean_text(item['text'])
    
    # 3. Create train/validation split
    print("\nCreating train/validation split...")
    X_train, X_val, y_train, y_val = create_train_validation_split(train_data)
    
    # 4. Extract test features if test data provided
    X_test = [item['text'] for item in test_data] if test_data else None
    
    # 5. Apply TF-IDF vectorization
    print("\nApplying TF-IDF vectorization...")
    vectorizer, X_train_tfidf = create_tfidf_pipeline(X_train, max_features)
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test) if X_test else None
    
    # 6. Prepare return dictionary
    result = {
        'X_train': X_train_tfidf,
        'X_val': X_val_tfidf,
        'X_test': X_test_tfidf,
        'y_train': y_train,
        'y_val': y_val,
        'vectorizer': vectorizer,
        'train_texts': X_train,
        'val_texts': X_val,
        'test_texts': X_test
    }
    
    print("\n=== Preprocessing Pipeline Complete ===")
    print(f"Training matrix shape: {X_train_tfidf.shape}")
    print(f"Validation matrix shape: {X_val_tfidf.shape}")
    if X_test_tfidf is not None:
        print(f"Test matrix shape: {X_test_tfidf.shape}")
    
    return result


def validate_preprocessing(original_data, processed_data):
    """
    Validate preprocessing pipeline integrity.
    
    Args:
        original_data (list): Original data before processing
        processed_data (list): Data after processing
    """
    print("=== Validating Preprocessing ===")
    
    # Check data integrity
    assert len(original_data) == len(processed_data), "Data length mismatch!"
    
    # Check label distribution is preserved
    original_labels = [item['label'] for item in original_data if item['label'] is not None]
    processed_labels = [item['label'] for item in processed_data if item['label'] is not None]
    
    original_distribution = Counter(original_labels)
    processed_distribution = Counter(processed_labels)
    
    assert original_distribution == processed_distribution, "Label distribution changed!"
    
    # Sample check: print before/after examples
    print("\nSample comparisons:")
    for i in range(min(3, len(original_data))):
        print(f"\nExample {i+1}:")
        print(f"Original: {original_data[i]['text'][:100]}...")
        print(f"Processed: {processed_data[i]['text'][:100]}...")
    
    print("\n✅ Preprocessing validation passed!")


def get_preprocessing_stats(data):
    """
    Get statistics about the preprocessed data.
    
    Args:
        data (dict): Preprocessed data dictionary from complete_preprocessing_pipeline
    """
    print("=== Preprocessing Statistics ===")
    
    # Training data stats
    train_texts = data['train_texts']
    val_texts = data['val_texts']
    
    train_lengths = [len(text.split()) for text in train_texts]
    val_lengths = [len(text.split()) for text in val_texts]
    
    print(f"\nTraining data:")
    print(f"  - Average text length: {np.mean(train_lengths):.1f} words")
    print(f"  - Median text length: {np.median(train_lengths):.1f} words")
    print(f"  - Max text length: {np.max(train_lengths)} words")
    print(f"  - Min text length: {np.min(train_lengths)} words")
    
    print(f"\nValidation data:")
    print(f"  - Average text length: {np.mean(val_lengths):.1f} words")
    print(f"  - Median text length: {np.median(val_lengths):.1f} words")
    print(f"  - Max text length: {np.max(val_lengths)} words")
    print(f"  - Min text length: {np.min(val_lengths)} words")
    
    # Label distribution
    train_dist = Counter(data['y_train'])
    val_dist = Counter(data['y_val'])
    
    print(f"\nLabel distribution:")
    print(f"  - Training: {dict(train_dist)}")
    print(f"  - Validation: {dict(val_dist)}")
    
    # TF-IDF stats
    print(f"\nTF-IDF vectorization:")
    print(f"  - Feature matrix shape: {data['X_train'].shape}")
    print(f"  - Sparsity: {(1 - data['X_train'].nnz / (data['X_train'].shape[0] * data['X_train'].shape[1])):.3f}")
