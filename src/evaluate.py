import os
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt

def evaluate_model(y_true, y_pred, y_prob=None, model_name="Model"):
    """
    Calculate Precision, Recall, F1-score, and PR-AUC.
    PR-AUC is the primary metric for highly imbalanced fraud datasets.
    """
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    pr_auc = None
    if y_prob is not None:
        # PR-AUC calculation
        pr_auc = average_precision_score(y_true, y_prob)

    metrics = {
        'Model': model_name,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'PR-AUC': pr_auc
    }
    
    return metrics

def print_results(metrics_list):
    """
    Print results cleanly to console for easy reading.
    """
    print("=" * 60)
    print(f"{'Model':<15} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'PR-AUC':<10}")
    print("-" * 60)
    for m in metrics_list:
        pr_auc_str = f"{m['PR-AUC']:.4f}" if m['PR-AUC'] is not None else "N/A"
        print(f"{m['Model']:<15} | {m['Precision']:<10.4f} | {m['Recall']:<10.4f} | {m['F1-Score']:<10.4f} | {pr_auc_str:<10}")
    print("=" * 60)

def save_results(metrics_list, filepath="outputs/results.csv"):
    """
    Save list of metric dictionaries to a CSV file.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    df = pd.DataFrame(metrics_list)
    df.to_csv(filepath, index=False)
    print(f"Results saved to {filepath}")

def plot_pr_curves(y_true, prob_dict, save_path="outputs/plots/pr_curves.png"):
    """
    Plot Precision-Recall curves for multiple models.
    prob_dict is a dictionary of {model_name: y_prob}
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    
    for model_name, y_prob in prob_dict.items():
        if y_prob is not None:
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            plt.plot(recall, precision, lw=2, label=f"{model_name}")
            
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve Comparison')
    plt.legend(loc="lower left")
    plt.grid(True)
    
    plt.savefig(save_path)
    print(f"PR-Curve plot saved to {save_path}")
    plt.show()
