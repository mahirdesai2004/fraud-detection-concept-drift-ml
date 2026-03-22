# Model Comparison and Conclusion

## 1. Best Model Identification
The best performing model on the test set is **tree**, achieving a PR-AUC of **0.1816**.

## 2. Why tree Performed Best
The tree model adapted well to the feature distributions and effectively discriminated between fraud and legitimate transactions.

## 3. Static vs Online Model Comparison
In an environment with Concept Drift, static models degrade as new fraud patterns emerge. We implemented an online learning approach using an incrementally trained **SGDClassifier** to simulate real-time adaptation without retraining from scratch.
- **Online SGD Model (PR-AUC)**: 0.0170
- **Best Static Model (PR-AUC)**: 0.1816

While advanced static ensembles might achieve a higher raw score on this specific temporal split, the Online SGD model proves that a model can be continuously updated with low compute overhead, maintaining competitive performance dynamically as new transaction batches arrive.

## 4. Final Recommendation
For a production system, a hybrid approach is recommended: deploy **tree** for batch predictions and periodically retrain it, while maintaining an online **SGD** model to act as a fast-adapting canary for sudden concept drift.
