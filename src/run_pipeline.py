import os
import pandas as pd
import matplotlib.pyplot as plt
from preprocess import load_data, preprocess_features, temporal_split_and_scale
from train import train_and_predict
from evaluate import evaluate_model, print_results, save_results, plot_pr_curves

def main():
    print("--- 1. Loading and Preprocessing Data ---")
    data_path = 'data/raw_data.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run download_data.py first.")
        return
        
    raw_df = load_data(data_path)
    df = preprocess_features(raw_df, time_col='timestamp', target_col='fraud_flag')
    
    print("\n--- 2. Splitting Data ---")
    X_train, X_val, X_test, y_train, y_val, y_test = temporal_split_and_scale(df, target_col='fraud_flag')
    
    print("\n--- 3. Training Models ---")
    models_to_train = ['logistic', 'tree', 'forest', 'xgboost', 'sgd']
    
    metrics_list = []
    prob_dict = {}
    
    for m in models_to_train:
        print(f"\nEvaluating {m}...")
        try:
            model, y_pred, y_prob = train_and_predict(m, X_train, y_train, X_val, y_val, X_test)
            metrics = evaluate_model(y_test, y_pred, y_prob, model_name=m)
            metrics_list.append(metrics)
            prob_dict[m] = y_prob
        except Exception as e:
            print(f"Failed to train {m}: {e}")
        
    print("\n--- 4. Evaluation Results ---")
    print_results(metrics_list)
    save_results(metrics_list, filepath="outputs/results.csv")
    
    print("\n--- 5. Generating Plots ---")
    # PR curves
    plot_pr_curves(y_test, prob_dict, save_path="outputs/plots/pr_curves.png")
    
    # Model Comparison Bar Chart
    metrics_df = pd.DataFrame(metrics_list)
    if not metrics_df.empty:
        plt.figure()
        metrics_df.set_index('Model')[['F1-Score', 'PR-AUC']].plot(kind='bar', figsize=(10, 6))
        plt.title('Model Comparison')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("outputs/plots/model_comparison.png", dpi=300)
        print("Model comparison plot saved to outputs/plots/model_comparison.png")
        plt.close('all')
        
    # Plot Fraud distribution
    plt.figure(figsize=(8, 5))
    df['fraud_flag'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
    plt.title('Fraud Distribution (Class Imbalance)')
    plt.xlabel('Legitimate (0) vs Fraud (1)')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("outputs/plots/fraud_distribution.png", dpi=300)
    print("Fraud distribution plot saved to outputs/plots/fraud_distribution.png")
    plt.close('all')

if __name__ == '__main__':
    main()
