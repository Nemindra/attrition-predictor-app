import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
# XGBoost is optional, TensorFlow/LSTM is removed for compatibility
try:
    import xgboost as xgb
except ImportError:
    xgb = None

def load_data(file_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load and preprocess data for training.
    Returns features (X) and target (y).
    """
    df = pd.read_csv(file_path)
    if 'attrition' not in df.columns:
        raise ValueError("Target column 'attrition' not found.")
    X = df.drop('attrition', axis=1)
    y = df['attrition']
    # Add more preprocessing as needed (encoding, scaling, etc.)
    return X, y

def train_models(X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """
    Train multiple ML models and log to MLflow.
    Returns a dict of model names to (model, metrics) tuples.
    """
    results = {}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Random Forest
    with mlflow.start_run(run_name="RandomForest"):
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        metrics = evaluate_model(y_test, y_pred)
        mlflow.sklearn.log_model(rf, "model")
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        results['RandomForest'] = (rf, metrics)

    # Logistic Regression
    with mlflow.start_run(run_name="LogisticRegression"):
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        metrics = evaluate_model(y_test, y_pred)
        mlflow.sklearn.log_model(lr, "model")
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        results['LogisticRegression'] = (lr, metrics)

    # XGBoost (if available)
    if xgb is not None:
        with mlflow.start_run(run_name="XGBoost"):
            xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            xgb_model.fit(X_train, y_train)
            y_pred = xgb_model.predict(X_test)
            metrics = evaluate_model(y_test, y_pred)
            mlflow.sklearn.log_model(xgb_model, "model")
            for k, v in metrics.items():
                mlflow.log_metric(k, v)
            results['XGBoost'] = (xgb_model, metrics)

    return results

def evaluate_model(y_true, y_pred) -> Dict[str, float]:
    """
    Calculate evaluation metrics for classification.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_pred)
    }

def register_best_model(results: Dict[str, Any], experiment_name: str = "AttritionModel"):
    """
    Register the best model in the MLflow Model Registry.
    """
    best_model_name = max(results, key=lambda k: results[k][1]["f1"])
    best_model, metrics = results[best_model_name]
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"Register_{best_model_name}"):
        mlflow.sklearn.log_model(best_model, "model", registered_model_name=experiment_name)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
    return best_model_name, metrics

# Example usage (not for production):
# X, y = load_data("/path/to/data.csv")
# results = train_models(X, y)
# best_model_name, best_metrics = register_best_model(results)
