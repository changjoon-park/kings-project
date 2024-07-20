# Train and evaluate the Linear SVM model
(
    svm_model,
    svm_accuracy,
    svm_precision,
    svm_recall,
    svm_f1,
    svm_roc_auc,
    svm_average_precision,
) = train_and_evaluate_model(
    models.get("Linear SVM"), X_train, y_train, X_test, y_test, "Linear SVM"
)

# Print the metrics
print(f"Linear SVM Metrics:")
print(f"  Accuracy: {svm_accuracy:.4f}")
print(f"  Precision: {svm_precision:.4f}")
print(f"  Recall: {svm_recall:.4f}")
print(f"  F1 Score: {svm_f1:.4f}")
print(f"  ROC AUC: {svm_roc_auc:.4f}")
print(f"  Average Precision: {svm_average_precision:.4f}")
