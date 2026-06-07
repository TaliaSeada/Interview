import os
import random
import zipfile
import json
import warnings

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores.*",
    category=UserWarning,
)

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


RANDOM_STATE = 42
N_SPLITS = 10
CLASS_NAMES = ["Not Survived", "Survived"]
THRESHOLD = 0.5
MAX_EPOCHS = 50
PATIENCE = 5
MIN_DELTA = 0.001
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 64


def set_seed(seed=RANDOM_STATE):
    """
    Set all available random seeds used by the training script.

    The function configures Python, NumPy, and PyTorch randomness so fold
    splits, model initialization, and shuffled mini-batches are reproducible
    across runs where the underlying libraries support deterministic behavior.

    Args:
        seed: Integer seed value shared across the training pipeline.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


class TitanicDataset(Dataset):
    """PyTorch dataset that serves preprocessed Titanic features and labels."""

    def __init__(self, X, y):
        """
        Convert feature and label arrays into tensors for model training.

        Args:
            X: Preprocessed feature matrix with one row per passenger.
            y: Binary survival labels as a pandas object or array-like values.
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        if hasattr(y, "to_numpy"):
            y = y.to_numpy()
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        """
        Return the number of passenger examples in the dataset.

        Returns:
            Number of rows available for batching.
        """
        return len(self.X)

    def __getitem__(self, idx):
        """
        Return one feature-label pair by row index.

        Args:
            idx: Integer position requested by a PyTorch DataLoader.

        Returns:
            Tuple containing the feature tensor and matching label tensor.
        """
        return self.X[idx], self.y[idx]


class TitanicModel(nn.Module):
    """Feed-forward binary classifier for Titanic survival prediction."""

    def __init__(self, input_dim):
        """
        Build the neural network architecture used for each CV fold.

        The network has three hidden layers with 64, 32, and 16 units. Each
        hidden layer uses batch normalization, ReLU activation, and dropout for
        regularization. The final layer returns one logit for binary
        classification.

        Args:
            input_dim: Number of columns produced by the fitted preprocessor.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        """
        Run a forward pass through the classifier.

        Args:
            x: Batch of preprocessed passenger features.

        Returns:
            Raw logits with shape ``(batch_size, 1)``.
        """
        return self.net(x)


def feature_engineering(df):
    """
    Create model features from the raw Titanic dataframe.

    The transformation fills missing ages by ``Sex`` and ``Pclass`` groups,
    creates cabin, title, and family-size features, fills missing embarked
    values, and removes raw identifier/text columns that are not passed into
    the sklearn preprocessor.

    Args:
        df: Raw Titanic rows containing the Kaggle competition columns.

    Returns:
        A dataframe with engineered columns ready for ``build_preprocessor``.
    """
    df = df.copy()
    # A more sophisticated approach could use passenger titles extracted from names.
    # However, to keep preprocessing simple and interpretable, age was imputed using demographic groups defined by Sex and Pclass.
    df['Age'] = df.groupby(['Sex', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))

    df['HasCabin'] = df['Cabin'].notna().astype(int)

    df.loc[df['Embarked'].isna(), 'Embarked'] = 'C'

    df['Title'] = df['Name'].str.extract(r',\s*([^\.]+)\.')
    # print('\nUnique Titles: \n', df['Title'].value_counts())
    df['Title'] = df['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})

    rare_titles = ['Dr', 'Rev', 'Major', 'Col', 'Don', 'Lady', 'Sir', 'Capt', 'the Countess', 'Jonkheer']
    df['Title'] = df['Title'].replace(rare_titles, 'Rare')
    # print('\nRare Replacement Titles: \n', df['Title'].value_counts())

    df['FamilySize'] = (df['SibSp'] + df['Parch'] + 1)

    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch'], axis=1)

    return df


def download_titanic():
    """
    Download and extract the Titanic Kaggle dataset when local data is absent.

    The function exits early if ``data/train.csv`` already exists. Otherwise,
    it expects the Kaggle CLI to be installed and authenticated, downloads the
    competition archive into ``data/``, and extracts the CSV files there.
    """
    if os.path.exists("data/train.csv"):
        return

    os.makedirs("data", exist_ok=True)

    os.system(
        "kaggle competitions download -c titanic -p data/"
    )

    with zipfile.ZipFile(
        "data/titanic.zip",
        "r"
    ) as zip_ref:
        zip_ref.extractall("data/")


def build_preprocessor():
    """
    Build the sklearn preprocessing pipeline for engineered Titanic features.

    Numeric features are standardized, categorical features
    are one-hot encoded, and already numeric
    categorical indicators are passed through unchanged.

    Returns:
        A ``ColumnTransformer`` that can be fitted on train folds and reused for
        validation or inference data.
    """
    numeric_features = [
        "Age", "Fare", "FamilySize"
    ]

    categorical_features = [
        "Sex", "Embarked", "Title"
    ]

    passthrough_features = [
        "HasCabin", "Pclass"
    ]

    numeric_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    return ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
        ("passthrough", "passthrough", passthrough_features),
    ])


def clone_state_dict(model):
    """
    Copy a model state dictionary independently of the live model parameters.

    Args:
        model: PyTorch model whose weights should be snapshotted.

    Returns:
        A CPU-based clone of each tensor in ``model.state_dict()``.
    """
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def predict_probabilities(model, X_processed):
    """
    Generate survival probabilities for preprocessed feature rows.

    Args:
        model: Trained ``TitanicModel`` instance.
        X_processed: Feature matrix already transformed by the fitted
            preprocessor.

    Returns:
        One-dimensional NumPy array of probabilities for the positive class.
    """
    model.eval()
    X_tensor = torch.tensor(X_processed, dtype=torch.float32)

    with torch.no_grad():
        logits = model(X_tensor)
        probabilities = torch.sigmoid(logits).cpu().numpy().flatten()

    return probabilities


def evaluate_predictions(y_true, probabilities):
    """
    Evaluate probability outputs at the configured decision threshold.

    Args:
        y_true: Ground-truth binary survival labels.
        probabilities: Predicted survival probabilities.

    Returns:
        Dictionary containing accuracy, precision, recall, F1, and the binary
        predictions used to compute those metrics.
    """
    predictions = (probabilities >= THRESHOLD).astype(int)

    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "predictions": predictions,
    }


def train_one_fold(fold, X_train_raw, y_train, X_val_raw, y_val):
    """
    Train and evaluate one stratified cross-validation fold.

    The fold owns its fitted feature engineering preprocessor and model. It
    trains with early stopping on validation accuracy, reloads the best epoch,
    computes validation probabilities and metrics, and returns the artifacts
    needed for fold reporting and best-fold selection.

    Args:
        fold: One-based fold number for logging and result records.
        X_train_raw: Raw passenger rows for the fold training split.
        y_train: Training labels aligned to ``X_train_raw``.
        X_val_raw: Raw passenger rows for the fold validation split.
        y_val: Validation labels aligned to ``X_val_raw``.

    Returns:
        Tuple of fold metrics, validation prediction rows, trained model, and
        fitted preprocessor.
    """
    set_seed()

    print("="*70)
    print(f"Fold {fold}/{N_SPLITS}")
    print("="*70)

    # we do this seperatly to prevent data leakage in age calculation
    X_train = feature_engineering(X_train_raw)
    X_val = feature_engineering(X_val_raw)

    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)

    print("Train class balance:")
    print(y_train.value_counts().sort_index())

    train_ds = TitanicDataset(X_train_processed, y_train)
    val_ds = TitanicDataset(X_val_processed, y_val)
    drop_last_train_batch = len(train_ds) % TRAIN_BATCH_SIZE == 1

    train_generator = torch.Generator().manual_seed(RANDOM_STATE)
    train_loader = DataLoader(
        train_ds,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        generator=train_generator,
        drop_last=drop_last_train_batch,
    )
    val_loader = DataLoader(val_ds, batch_size=VAL_BATCH_SIZE, shuffle=False)
    print("Train size: ", X_train_processed.shape)
    print("Validation size: ", X_val_processed.shape )

    if drop_last_train_batch:
        print("Dropping the final 1-row training batch so BatchNorm can compute batch statistics.")

    model = TitanicModel(input_dim=X_train_processed.shape[1])
    print(model)
    
    # binary cross entropy with sigmoid
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_val_acc = 0.0
    best_epoch = 0
    best_state = clone_state_dict(model)
    epochs_without_improvement = 0
    epochs_trained = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        train_loss = 0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        val_acc = correct / total
        epochs_trained = epoch + 1
        print(
            f"Fold {fold} | Epoch {epoch+1:03d} | "
            f"Loss: {train_loss/len(train_loader):.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc + MIN_DELTA:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_state = clone_state_dict(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(
                f"Early stopping fold {fold} at epoch {epoch+1:03d}. "
                f"Best Val Acc: {best_val_acc:.4f} at epoch {best_epoch:03d}."
            )
            break

    model.load_state_dict(best_state)
    probabilities = predict_probabilities(model, X_val_processed)
    metrics = evaluate_predictions(y_val, probabilities)

    fold_result = {
        "fold": fold,
        "train_rows": len(X_train_raw),
        "validation_rows": len(X_val_raw),
        "features": X_train_processed.shape[1],
        "epochs_trained": epochs_trained,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
    }

    fold_predictions = pd.DataFrame(
        {
            "fold": fold,
            "row_index": X_val_raw.index,
            "actual": y_val.to_numpy(),
            "predicted": metrics["predictions"],
            "survival_probability": probabilities,
        }
    )

    return fold_result, fold_predictions, model, preprocessor


def summarize_cv_results(cv_results):
    """
    Compute mean and standard deviation for core CV metrics.

    Args:
        cv_results: Dataframe with one row per fold and metric columns.

    Returns:
        Dictionary with ``mean_*`` and ``std_*`` entries for accuracy,
        precision, recall, and F1.
    """
    metrics = ["accuracy", "precision", "recall", "f1"]
    summary = {}

    for metric in metrics:
        summary[f"mean_{metric}"] = cv_results[metric].mean()
        summary[f"std_{metric}"] = cv_results[metric].std()

    return summary


def run_cross_validation(df):
    """
    Run stratified k-fold training and collect all model artifacts.

    Args:
        df: Titanic training dataframe containing the ``Survived`` target.

    Returns:
        Dictionary with aggregate summary metrics, per-fold metrics,
        out-of-fold predictions, the best model, its fitted preprocessor, and
        the selected best-fold result.
    """
    print("="*70)
    print("Training TitanicModel")
    print("="*70)

    X = df.drop(["Survived"], axis=1)
    y = df["Survived"]

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_results = []
    prediction_frames = []
    best_fold_result = None
    best_model = None
    best_preprocessor = None

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        fold_result, fold_predictions, model, preprocessor = train_one_fold(
            fold=fold,
            X_train_raw=X.iloc[train_idx],
            y_train=y.iloc[train_idx],
            X_val_raw=X.iloc[val_idx],
            y_val=y.iloc[val_idx],
        )
        fold_results.append(fold_result)
        prediction_frames.append(fold_predictions)

        if (
            best_fold_result is None
            or fold_result["f1"] > best_fold_result["f1"]
            or (
                fold_result["f1"] == best_fold_result["f1"]
                and fold_result["accuracy"] > best_fold_result["accuracy"]
            )
        ):
            best_fold_result = fold_result
            best_model = model
            best_preprocessor = preprocessor

    cv_results = pd.DataFrame(fold_results)
    cv_predictions = pd.concat(prediction_frames, ignore_index=True)
    summary = {
        "best_fold": int(best_fold_result["fold"]),
        "best_fold_f1": float(best_fold_result["f1"]),
        "best_fold_accuracy": float(best_fold_result["accuracy"]),
    }
    summary.update(summarize_cv_results(cv_results))

    return {
        "summary": summary,
        "cv_results": cv_results,
        "cv_predictions": cv_predictions,
        "best_model": best_model,
        "best_preprocessor": best_preprocessor,
        "best_fold_result": best_fold_result,
    }


def save_training_outputs(training_run, output_dir="model"):
    """
    Persist cross-validation results, best-fold artifacts, and metadata.

    Args:
        training_run: Output dictionary produced by ``run_cross_validation``.
        output_dir: Directory where CSV reports, model weights, preprocessor,
            and metadata JSON should be written.
    """
    os.makedirs(output_dir, exist_ok=True)

    cv_results = training_run["cv_results"]
    cv_predictions = training_run["cv_predictions"]
    best_model = training_run["best_model"]
    best_preprocessor = training_run["best_preprocessor"]
    best_fold_result = training_run["best_fold_result"]
    summary = training_run["summary"]

    cv_results.to_csv(f"{output_dir}/cv_results.csv", index=False)
    cv_predictions.to_csv(f"{output_dir}/cv_predictions.csv", index=False)

    torch.save(best_model.state_dict(), f"{output_dir}/model.pt")
    joblib.dump(best_preprocessor, f"{output_dir}/preprocessor.pkl")

    metadata = {
        "random_state": RANDOM_STATE,
        "n_splits": N_SPLITS,
        "threshold": THRESHOLD,
        "best_fold": int(best_fold_result["fold"]),
        "selection_metric": "f1",
        "best_fold_f1": float(best_fold_result["f1"]),
        "best_fold_accuracy": float(best_fold_result["accuracy"]),
        "mean_f1": float(summary["mean_f1"]),
        "mean_accuracy": float(summary["mean_accuracy"]),
    }
    with open(f"{output_dir}/cv_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("="*70)
    print("Cross-validation results:")
    print(cv_results.to_string(index=False))
    print()
    print("Mean metrics:")
    print(cv_results[["accuracy", "precision", "recall", "f1"]].mean().to_string())
    print()
    print("Std metrics:")
    print(cv_results[["accuracy", "precision", "recall", "f1"]].std().to_string())
    print("="*70)
    print(
        f"Saved best fold artifacts to {output_dir}/model.pt "
        f"and {output_dir}/preprocessor.pkl "
        f"(fold {metadata['best_fold']})."
    )
    print(f"Saved CV reports to {output_dir}/cv_results.csv and {output_dir}/cv_predictions.csv.")
    print_overall_report(cv_predictions)


def print_overall_report(cv_predictions):
    """
    Print an overall out-of-fold classification report.

    Args:
        cv_predictions: Dataframe containing ``actual`` and ``predicted``
            columns for all validation rows across folds.
    """
    y_true = cv_predictions["actual"]
    y_pred = cv_predictions["predicted"]

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_df = pd.DataFrame(
        cm,
        index=[f"True: {name}" for name in CLASS_NAMES],
        columns=[f"Predicted: {name}" for name in CLASS_NAMES],
    )

    print("="*70)
    print("Overall out-of-fold evaluation:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    print("Confusion Matrix:")
    print(cm_df)
    print("="*70)


def main():
    """
    Execute the full training workflow from data loading to artifact saving.

    The script loads the local Titanic training CSV, runs stratified
    cross-validation, and writes the best fold model, fitted preprocessor,
    fold metrics, out-of-fold predictions, and metadata under ``model/``.
    """
    set_seed()

    print("="*70)
    print("Loading data..")
    download_titanic()
    df = pd.read_csv('data/train.csv')
    print(df.head())
    print("="*70)

    training_run = run_cross_validation(df)
    save_training_outputs(training_run)


if __name__ == "__main__":
    main()
