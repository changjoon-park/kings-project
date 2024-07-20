## Parameters: XGBoost
xgb_parameters = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 4,
    "min_child_weight": 3,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
}

## Parameters: Linear SVM
svm_parameters = {
    "max_iter": 50000,
}

## Update specific models with desired parameters
updated_models = []
for model in models:
    if model.name == "XGBoost":
        updated_model = Model(model.name, xgboost(**xgb_parameters))
    elif model.name == "Linear SVM":
        updated_model = Model(
            model.name, linear_support_vector_machine(**svm_parameters)
        )
    else:
        updated_model = model
    updated_models.append(updated_model)

for model in updated_models:
    model.train_model(X_train, y_train)
    (
        model,
        accuracy,
        precision,
        recall,
        f1,
        cm,
        roc_auc,
        fpr,
        tpr,
        precision_curve,
        recall_curve,
        average_precision,
    ) = model.evaluate_model(X_test, y_test)

    print_metrics(
        model, accuracy, precision, recall, f1, cm, roc_auc, average_precision
    )
    plot_results(
        model, fpr, tpr, roc_auc, precision_curve, recall_curve, average_precision, cm
    )
