"""
Model Evaluation Module for Fake News Classification Project
Pure utility module for storing, retrieving, and visualizing model results.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path

# Presentation styling constants
PRESENTATION_STYLE = {
    "figsize": (10, 6),
    "colors": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"],
    "font_size": 12,
    "title_size": 14,
    "save_dpi": 300
}

def save_model_results(model_name, display_name, accuracy, training_time_minutes, 
                      model_architecture=None, preprocessing_type=None, 
                      hyperparameters=None, dataset_info=None, 
                      train_accuracy=None, test_accuracy=None, 
                      confusion_matrix=None, classification_report=None,
                      results_dir="results"):
    """
    Save model results using keyword arguments for cleaner interface.
    
    Args:
        model_name (str): Unique identifier for the model (e.g., "baseline_lr")
        display_name (str): Human-readable model name (e.g., "Baseline LogisticRegression")
        accuracy (float): Main accuracy score as decimal (0-1)
        training_time_minutes (float): Training time in minutes
        model_architecture (str, optional): Model architecture summary
        preprocessing_type (str, optional): Type of preprocessing used
        hyperparameters (dict, optional): Model hyperparameters
        dataset_info (dict, optional): Dataset statistics
        train_accuracy (float, optional): Training accuracy as decimal (0-1)
        test_accuracy (float, optional): Test accuracy as decimal (0-1)
        confusion_matrix (list, optional): 2x2 confusion matrix [[tp, fp], [fn, tn]]
        classification_report (dict, optional): Classification report from sklearn
        results_dir (str): Directory to save results
    
    Example:
        save_model_results(
            model_name="baseline_lr",
            display_name="Baseline LogisticRegression", 
            accuracy=0.9290,
            training_time_minutes=1.5,
            model_architecture="LogisticRegression with CountVectorizer",
            preprocessing_type="standardized_clean_text",
            hyperparameters={"max_iter": 1000, "random_state": 42},
            dataset_info={"training_samples": 27320, "validation_samples": 6831}
        )
    """
    # Create results directory if it doesn't exist
    Path(results_dir).mkdir(exist_ok=True)
    
    # Build results dictionary from arguments
    results_dict = {
        "display_name": display_name,
        "accuracy": accuracy,
        "training_time_minutes": training_time_minutes,
        "timestamp": datetime.now().isoformat(),
        "model_id": model_name
    }
    
    # Add optional fields if provided
    if model_architecture is not None:
        results_dict["model_architecture"] = model_architecture
    if preprocessing_type is not None:
        results_dict["preprocessing_type"] = preprocessing_type
    if hyperparameters is not None:
        results_dict["hyperparameters"] = hyperparameters
    if dataset_info is not None:
        results_dict["dataset_info"] = dataset_info
    if train_accuracy is not None:
        results_dict["train_accuracy"] = train_accuracy
    if test_accuracy is not None:
        results_dict["test_accuracy"] = test_accuracy
    if confusion_matrix is not None:
        results_dict["confusion_matrix"] = confusion_matrix
    if classification_report is not None:
        results_dict["classification_report"] = classification_report
    
    # Save to JSON file
    filename = f"{results_dir}/{model_name}_results.json"
    with open(filename, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"Results saved to {filename}")
    print(f"Model: {display_name}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Training Time: {training_time_minutes:.2f} minutes")

def load_all_model_results(results_dir="results"):
    """
    Load all saved model results from JSON files.
    
    Args:
        results_dir (str): Directory containing result files
        
    Returns:
        dict: Dictionary with model IDs as keys and results as values
    """
    results = {}
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Results directory {results_dir} not found. No results to load.")
        return results
    
    for file_path in results_path.glob("*_results.json"):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                model_id = data.get("model_id", file_path.stem.replace("_results", ""))
                results[model_id] = data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    print(f"Loaded {len(results)} model results")
    return results

def get_model_names_and_accuracies(models_dict):
    """
    Extract model names and accuracies for plotting.
    
    Args:
        models_dict (dict): Dictionary of model results
        
    Returns:
        tuple: (model_names, accuracies_percent)
    """
    model_names = []
    accuracies = []
    
    for model_id, data in models_dict.items():
        display_name = data.get("display_name", model_id)
        accuracy = data.get("accuracy", 0)
        
        model_names.append(display_name)
        accuracies.append(accuracy * 100)  # Convert to percentage
    
    return model_names, accuracies

def plot_best_models_comparison(models_dict=None, save_path="presentation_assets/executive_summary.png", title="Model Performance Comparison", chart_type="line", order=None):
    """
    Create executive summary chart comparing models with improved scaling.
    
    Args:
        models_dict (dict): Dictionary of model results. If None, loads from files.
        save_path (str): Path to save the chart
        title (str): Chart title
        chart_type (str): "line" or "bar" chart type
        order (list): Optional list of model IDs to specify the order of models in the chart.
                     If None, models will be displayed in the order they appear in the dictionary.
                     Example: ["baseline_lr", "glove_pooling_lr", "full_bert_finetuned"]
    """
    if models_dict is None:
        models_dict = load_all_model_results()
    
    if not models_dict:
        print("No model results found. Please save model results first.")
        return
    
    # If order is specified, reorder the models_dict
    if order is not None:
        # Filter and reorder models_dict based on the specified order
        ordered_models_dict = {}
        for model_id in order:
            if model_id in models_dict:
                ordered_models_dict[model_id] = models_dict[model_id]
            else:
                print(f"Warning: Model '{model_id}' not found in results. Available models: {list(models_dict.keys())}")
        
        # Add any remaining models not specified in order
        for model_id, data in models_dict.items():
            if model_id not in ordered_models_dict:
                ordered_models_dict[model_id] = data
        
        models_dict = ordered_models_dict
    
    model_names, accuracies = get_model_names_and_accuracies(models_dict)
    
    # Calculate appropriate y-axis limits for better visualization
    min_acc = min(accuracies)
    max_acc = max(accuracies)
    acc_range = max_acc - min_acc
    
    # If the range is small (< 10%), use a focused scale
    if acc_range < 10:
        # Add some padding (10% of range or minimum 2 percentage points)
        padding = max(acc_range * 0.1, 2)
        y_min = max(0, min_acc - padding)
        y_max = min(100, max_acc + padding)
        
        # Ensure minimum visible range of 5 percentage points
        if y_max - y_min < 5:
            center = (y_min + y_max) / 2
            y_min = max(0, center - 2.5)
            y_max = min(100, center + 2.5)
    else:
        # For larger ranges, use the standard 0-105 scale
        y_min = 0
        y_max = 105
    
    # Create the chart
    plt.figure(figsize=PRESENTATION_STYLE["figsize"])
    
    if chart_type == "line":
        # Line chart
        plt.plot(range(len(model_names)), accuracies, 
                marker='o', markersize=8, linewidth=2, 
                color=PRESENTATION_STYLE["colors"][0])
        
        # Add value labels on points
        for i, (name, acc) in enumerate(zip(model_names, accuracies)):
            # Adjust label position based on y-axis scale
            label_offset = (y_max - y_min) * 0.02  # 2% of visible range
            plt.text(i, acc + label_offset, f'{acc:.1f}%', ha='center', va='bottom', 
                    fontsize=PRESENTATION_STYLE["font_size"], fontweight='bold')
        
        plt.xticks(range(len(model_names)), model_names)
    else:
        # Bar chart (original)
        bars = plt.bar(model_names, accuracies, color=PRESENTATION_STYLE["colors"][:len(model_names)])
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            label_offset = (y_max - y_min) * 0.01  # 1% of visible range
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + label_offset, 
                    f'{acc:.1f}%', ha='center', va='bottom', 
                    fontsize=PRESENTATION_STYLE["font_size"], fontweight='bold')
    
    plt.title(title, fontsize=PRESENTATION_STYLE["title_size"], fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=PRESENTATION_STYLE["font_size"])
    plt.xlabel('Models', fontsize=PRESENTATION_STYLE["font_size"])
    
    # Use the calculated y-axis limits
    plt.ylim(y_min, y_max)
    
    # Add grid with appropriate spacing
    plt.grid(axis='y', alpha=0.3)
    
    # Add a text annotation showing the scale if using focused view
    if acc_range < 10:
        plt.text(0.02, 0.98, f"Focused scale: {y_min:.1f}% - {y_max:.1f}%", 
                transform=plt.gca().transAxes, fontsize=10, 
                verticalalignment='top', alpha=0.7,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Rotate x-axis labels if many models
    if len(model_names) > 3:
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save the chart
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=PRESENTATION_STYLE["save_dpi"], bbox_inches='tight')
    plt.show()
    print(f"Chart saved to {save_path}")
    print(f"Y-axis range: {y_min:.1f}% - {y_max:.1f}% (Range: {y_max-y_min:.1f} percentage points)")

def plot_model_accuracy_comparison(models_dict=None, save_path="presentation_assets/model_comparison.png", order=None):
    """
    Create detailed model comparison chart with accuracy and training time (improved scaling).
    
    Args:
        models_dict (dict): Dictionary of model results. If None, loads from files.
        save_path (str): Path to save the chart
        order (list): Optional list of model IDs to specify the order of models in the chart.
                     If None, models will be displayed in the order they appear in the dictionary.
                     Example: ["baseline_lr", "glove_pooling_lr", "full_bert_finetuned"]
    """
    if models_dict is None:
        models_dict = load_all_model_results()
    
    if not models_dict:
        print("No model results found. Please save model results first.")
        return
    
    # If order is specified, reorder the models_dict
    if order is not None:
        # Filter and reorder models_dict based on the specified order
        ordered_models_dict = {}
        for model_id in order:
            if model_id in models_dict:
                ordered_models_dict[model_id] = models_dict[model_id]
            else:
                print(f"Warning: Model '{model_id}' not found in results. Available models: {list(models_dict.keys())}")
        
        # Add any remaining models not specified in order
        for model_id, data in models_dict.items():
            if model_id not in ordered_models_dict:
                ordered_models_dict[model_id] = data
        
        models_dict = ordered_models_dict
    
    # Extract data
    model_names, accuracies = get_model_names_and_accuracies(models_dict)
    training_times = []
    
    for model_id, data in models_dict.items():
        training_times.append(data.get("training_time_minutes", 0))
    
    # Calculate appropriate y-axis limits for accuracy plot
    min_acc = min(accuracies)
    max_acc = max(accuracies)
    acc_range = max_acc - min_acc
    
    # If the range is small (< 10%), use a focused scale
    if acc_range < 10:
        # Add some padding (10% of range or minimum 2 percentage points)
        padding = max(acc_range * 0.1, 2)
        y_min = max(0, min_acc - padding)
        y_max = min(100, max_acc + padding)
        
        # Ensure minimum visible range of 5 percentage points
        if y_max - y_min < 5:
            center = (y_min + y_max) / 2
            y_min = max(0, center - 2.5)
            y_max = min(100, center + 2.5)
    else:
        # For larger ranges, use the standard 0-105 scale
        y_min = 0
        y_max = 105
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy comparison with improved scaling
    bars1 = ax1.bar(model_names, accuracies, color=PRESENTATION_STYLE["colors"][:len(model_names)])
    for bar, acc in zip(bars1, accuracies):
        label_offset = (y_max - y_min) * 0.01  # 1% of visible range
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + label_offset, 
                f'{acc:.1f}%', ha='center', va='bottom', 
                fontsize=PRESENTATION_STYLE["font_size"], fontweight='bold')
    
    ax1.set_title('Model Accuracy Comparison', fontsize=PRESENTATION_STYLE["title_size"], fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=PRESENTATION_STYLE["font_size"])
    ax1.set_ylim(y_min, y_max)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add scale annotation if using focused view
    if acc_range < 10:
        ax1.text(0.02, 0.98, f"Focused scale: {y_min:.1f}% - {y_max:.1f}%", 
                transform=ax1.transAxes, fontsize=9, 
                verticalalignment='top', alpha=0.7,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    if len(model_names) > 3:
        ax1.tick_params(axis='x', rotation=45)
    
    # Training time comparison (unchanged)
    bars2 = ax2.bar(model_names, training_times, color=PRESENTATION_STYLE["colors"][:len(model_names)])
    for bar, time_val in zip(bars2, training_times):
        if time_val > 0:  # Only show label if time is recorded
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(training_times) * 0.01, 
                    f'{time_val:.1f}m', ha='center', va='bottom', 
                    fontsize=PRESENTATION_STYLE["font_size"], fontweight='bold')
    
    ax2.set_title('Training Time Comparison', fontsize=PRESENTATION_STYLE["title_size"], fontweight='bold')
    ax2.set_ylabel('Training Time (minutes)', fontsize=PRESENTATION_STYLE["font_size"])
    ax2.grid(axis='y', alpha=0.3)
    
    if len(model_names) > 3:
        ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save the chart
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=PRESENTATION_STYLE["save_dpi"], bbox_inches='tight')
    plt.show()
    print(f"Chart saved to {save_path}")
    print(f"Accuracy y-axis range: {y_min:.1f}% - {y_max:.1f}% (Range: {y_max-y_min:.1f} percentage points)")

def plot_train_val_test_comparison(model_id, models_dict=None, save_path=None):
    """
    Create chart comparing training, validation, and test accuracy for a specific model.
    
    Args:
        model_id (str): ID of the model to analyze
        models_dict (dict): Dictionary of model results. If None, loads from files.
        save_path (str): Path to save the chart. If None, uses default naming.
    """
    if models_dict is None:
        models_dict = load_all_model_results()
    
    if model_id not in models_dict:
        print(f"Model '{model_id}' not found in results.")
        available_models = list(models_dict.keys())
        print(f"Available models: {available_models}")
        return
    
    data = models_dict[model_id]
    
    # Extract accuracies
    train_acc = data.get("train_accuracy", 0) * 100
    val_acc = data.get("accuracy", 0) * 100  # Main accuracy (usually validation)
    test_acc = data.get("test_accuracy", val_acc) * 100  # Use val if test not available
    
    categories = ['Training', 'Validation', 'Test']
    accuracies = [train_acc, val_acc, test_acc]
    
    # Create the chart
    plt.figure(figsize=(8, 6))
    bars = plt.bar(categories, accuracies, color=PRESENTATION_STYLE["colors"][:3])
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        if acc > 0:  # Only show label if accuracy is recorded
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{acc:.1f}%', ha='center', va='bottom', 
                    fontsize=PRESENTATION_STYLE["font_size"], fontweight='bold')
    
    display_name = data.get("display_name", model_id)
    plt.title(f'{display_name} - Training vs Validation vs Test', 
              fontsize=PRESENTATION_STYLE["title_size"], fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=PRESENTATION_STYLE["font_size"])
    plt.ylim(0, 105)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save the chart
    if save_path is None:
        save_path = f"presentation_assets/{model_id}_train_val_test.png"
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=PRESENTATION_STYLE["save_dpi"], bbox_inches='tight')
    plt.show()
    print(f"Chart saved to {save_path}")

def generate_presentation_summary(models_dict=None):
    """
    Generate text summary for presentation slides.
    
    Args:
        models_dict (dict): Dictionary of model results. If None, loads from files.
        
    Returns:
        dict: Summary statistics and text for presentation
    """
    if models_dict is None:
        models_dict = load_all_model_results()
    
    if not models_dict:
        return {"error": "No model results found"}
    
    # Find best model
    best_model = max(models_dict.items(), key=lambda x: x[1].get("accuracy", 0))
    best_id, best_data = best_model
    
    # Calculate improvements
    accuracies = [data.get("accuracy", 0) for data in models_dict.values()]
    min_acc, max_acc = min(accuracies), max(accuracies)
    improvement = (max_acc - min_acc) * 100
    
    # Generate summary
    summary = {
        "best_model_name": best_data.get("display_name", best_id),
        "best_accuracy": f"{best_data.get('accuracy', 0) * 100:.2f}%",
        "total_models_tested": len(models_dict),
        "accuracy_improvement": f"{improvement:.1f} percentage points",
        "best_architecture": best_data.get("architecture", "Unknown"),
        "training_time": f"{best_data.get('training_time_minutes', 0):.1f} minutes",
        "model_ranking": sorted(models_dict.items(), 
                               key=lambda x: x[1].get("accuracy", 0), 
                               reverse=True)
    }
    
    return summary

def create_model_performance_table(models_dict=None, format_type="markdown", order=None):
    """
    Create a formatted table of model performance for README or presentation.
    
    Args:
        models_dict (dict): Dictionary of model results. If None, loads from files.
        format_type (str): "markdown" or "html"
        order (list): Optional list of model IDs to specify the order of models in the table.
                     If None, models will be sorted by accuracy (descending).
                     Example: ["baseline_lr", "glove_pooling_lr", "full_bert_finetuned"]
        
    Returns:
        str: Formatted table string
    """
    if models_dict is None:
        models_dict = load_all_model_results()
    
    if not models_dict:
        return "No model results found"
    
    # If order is specified, use that order; otherwise sort by accuracy
    if order is not None:
        # Filter and reorder models_dict based on the specified order
        ordered_models_dict = {}
        for model_id in order:
            if model_id in models_dict:
                ordered_models_dict[model_id] = models_dict[model_id]
            else:
                print(f"Warning: Model '{model_id}' not found in results. Available models: {list(models_dict.keys())}")
        
        # Add any remaining models not specified in order
        for model_id, data in models_dict.items():
            if model_id not in ordered_models_dict:
                ordered_models_dict[model_id] = data
        
        sorted_models = list(ordered_models_dict.items())
    else:
        # Sort by accuracy (descending) - default behavior
        sorted_models = sorted(models_dict.items(), 
                             key=lambda x: x[1].get("accuracy", 0), 
                             reverse=True)
    
    if format_type == "markdown":
        # Create markdown table
        header = "| Model | Accuracy | Training Time | Architecture/Setup |\n"
        separator = "|-------|----------|---------------|-------------|\n"
        rows = ""
        
        for model_id, data in sorted_models:
            display_name = data.get("display_name", model_id)
            accuracy = f"{data.get('accuracy', 0) * 100:.2f}%"
            time_str = f"{data.get('training_time_minutes', 0):.1f} min"
            architecture = data.get("model_architecture", "Unknown")
            
            rows += f"| {display_name} | {accuracy} | {time_str} | {architecture} |\n"
        
        return header + separator + rows
    
    # Add HTML format if needed
    return "HTML format not implemented yet"

def list_available_models(results_dir="results"):
    """
    List all available model results.
    
    Args:
        results_dir (str): Directory containing result files
        
    Returns:
        list: List of available model IDs
    """
    models_dict = load_all_model_results(results_dir)
    
    if not models_dict:
        print("No model results found.")
        return []
    
    print("Available models:")
    for model_id, data in models_dict.items():
        display_name = data.get("display_name", model_id)
        accuracy = data.get("accuracy", 0) * 100
        print(f"  {model_id}: {display_name} ({accuracy:.2f}%)")
    
    return list(models_dict.keys())
