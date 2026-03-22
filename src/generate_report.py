import pandas as pd
import os

def generate_report(results_path='outputs/results.csv', report_path='outputs/model_explanation.md'):
    if not os.path.exists(results_path):
        print("Results CSV not found. Run pipeline first.")
        return
        
    df = pd.read_csv(results_path)
    
    if 'PR-AUC' in df.columns and not df['PR-AUC'].isnull().all():
        best_model_row = df.loc[df['PR-AUC'].idxmax()]
        main_metric = 'PR-AUC'
    else:
        best_model_row = df.loc[df['F1-Score'].idxmax()]
        main_metric = 'F1-Score'
        
    best_model = best_model_row['Model']
    best_score = best_model_row[main_metric]
    sgd_row = df[df['Model'] == 'sgd'].iloc[0] if len(df[df['Model'] == 'sgd']) > 0 else None
    
    md_content = f"# Model Comparison and Conclusion\n\n"
    md_content += f"## 1. Best Model Identification\n"
    md_content += f"The best performing model on the test set is **{best_model}**, achieving a {main_metric} of **{best_score:.4f}**.\n\n"
    
    md_content += f"## 2. Why {best_model} Performed Best\n"
    if best_model in ['xgboost', 'forest']:
        md_content += f"Ensemble tree-based models like {best_model} naturally capture complex, non-linear feature interactions and are generally robust to outliers and skewed distributions. In a highly imbalanced dataset with potential concept drift, their ability to partition the feature space dynamically makes them superior to simpler linear models.\n\n"
    elif best_model in ['logistic', 'sgd']:
        md_content += f"Linear models like {best_model} performed exceptionally well, likely due to the dataset being cleanly separable or the high effectiveness of the added temporal preprocessing features (`is_night`, `log_amount`).\n\n"
    else:
        md_content += f"The {best_model} model adapted well to the feature distributions and effectively discriminated between fraud and legitimate transactions.\n\n"
        
    md_content += f"## 3. Static vs Online Model Comparison\n"
    md_content += f"In an environment with Concept Drift, static models degrade as new fraud patterns emerge. We implemented an online learning approach using an incrementally trained **SGDClassifier** to simulate real-time adaptation without retraining from scratch.\n"
    
    if sgd_row is not None:
        md_content += f"- **Online SGD Model ({main_metric})**: {sgd_row[main_metric]:.4f}\n"
        md_content += f"- **Best Static Model ({main_metric})**: {best_score:.4f}\n\n"
        md_content += "While advanced static ensembles might achieve a higher raw score on this specific temporal split, the Online SGD model proves that a model can be continuously updated with low compute overhead, maintaining competitive performance dynamically as new transaction batches arrive.\n\n"
        
    md_content += "## 4. Final Recommendation\n"
    md_content += f"For a production system, a hybrid approach is recommended: deploy **{best_model}** for batch predictions and periodically retrain it, while maintaining an online **SGD** model to act as a fast-adapting canary for sudden concept drift.\n"
    
    with open(report_path, 'w') as f:
        f.write(md_content)
        
    print(f"Report generated successfully at {report_path}")

if __name__ == '__main__':
    generate_report()
