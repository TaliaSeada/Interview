import os
from pathlib import Path

import joblib

os.environ.setdefault("MPLCONFIGDIR", ".matplotlib-cache")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from train import TitanicModel, feature_engineering


CLASS_NAMES = ["Not Survived", "Survived"]
LABEL_MAP = {0: "Not Survived", 1: "Survived"}


@st.cache_data
def load_dataset(data_path):
    """
    Load the CSV selected in the inference sidebar.

    Args:
        data_path: Path to the dataset CSV.

    Returns:
        Dataframe containing rows to score, with optional ``Survived`` labels.
    """
    return pd.read_csv(data_path)


@st.cache_resource
def load_preprocessor(preprocessor_path):
    """
    Load the fitted sklearn preprocessor saved during training.

    Args:
        preprocessor_path: Path to ``preprocessor.pkl``.

    Returns:
        Fitted preprocessing object used to transform engineered features.
    """
    return joblib.load(preprocessor_path)


@st.cache_resource
def load_model(model_path, input_dim):
    """
    Reconstruct the Titanic model and load trained weights.

    Args:
        model_path: Path to the saved PyTorch state dictionary.
        input_dim: Number of transformed input features expected by the model.

    Returns:
        ``TitanicModel`` set to evaluation mode on CPU.
    """
    model = TitanicModel(input_dim=input_dim)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_probabilities(model, X_processed):
    """
    Generate survival probabilities for transformed inference rows.

    Args:
        model: Loaded ``TitanicModel`` instance.
        X_processed: Feature matrix produced by the fitted preprocessor.

    Returns:
        One-dimensional NumPy array of survival probabilities.
    """
    X_tensor = torch.tensor(X_processed, dtype=torch.float32)

    with torch.no_grad():
        logits = model(X_tensor)
        probabilities = torch.sigmoid(logits).cpu().numpy().flatten()

    return probabilities


def apply_threshold(probabilities, threshold):
    """
    Convert survival probabilities into binary predictions.

    Args:
        probabilities: Predicted probability for class ``1``.
        threshold: Probability cutoff used to mark a passenger as survived.

    Returns:
        NumPy array of integer predictions.
    """
    return (probabilities >= threshold).astype(int)


def metric_summary(y_true, y_pred):
    """
    Calculate evaluation metrics when the input CSV includes labels.

    Args:
        y_true: Ground-truth binary survival labels.
        y_pred: Binary predictions produced by the current threshold.

    Returns:
        Dictionary containing accuracy, precision, recall, and F1.
    """
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }


def labeled_confusion_matrix(y_true, y_pred):
    """
    Build a confusion matrix dataframe with readable survival class labels.

    Args:
        y_true: Ground-truth binary survival labels.
        y_pred: Binary predictions.

    Returns:
        Two-by-two dataframe indexed and labeled with class names.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return pd.DataFrame(
        cm,
        index=[f"True: {name}" for name in CLASS_NAMES],
        columns=[f"Predicted: {name}" for name in CLASS_NAMES],
    )


def show_metrics(metrics):
    """
    Render four evaluation metrics in Streamlit columns.

    Args:
        metrics: Mapping from metric names to numeric values.
    """
    cols = st.columns(4)
    for col, (name, value) in zip(cols, metrics.items()):
        col.metric(name, f"{value:.3f}")


def prediction_summary(results):
    """
    Summarize the scored inference results for the top-level metric row.

    Args:
        results: Prediction table returned by ``build_prediction_table``.

    Returns:
        Dictionary with row count, class counts, and average survival
        probability.
    """
    survived_count = int((results["Predicted"] == 1).sum())
    not_survived_count = int((results["Predicted"] == 0).sum())
    total_rows = len(results)

    return {
        "Rows scored": total_rows,
        "Predicted survived": survived_count,
        "Predicted not survived": not_survived_count,
        "Avg survival probability": float(results["Survival Probability"].mean()),
    }


def show_prediction_summary(summary):
    """
    Render aggregate inference counts and mean probability in Streamlit.

    Args:
        summary: Dictionary produced by ``prediction_summary``.
    """
    cols = st.columns(4)
    cols[0].metric("Rows scored", f"{summary['Rows scored']:,}")
    cols[1].metric("Predicted survived", f"{summary['Predicted survived']:,}")
    cols[2].metric("Predicted not survived", f"{summary['Predicted not survived']:,}")
    cols[3].metric(
        "Avg survival probability",
        f"{summary['Avg survival probability']:.3f}",
    )


def plot_confusion_matrix(cm_df):
    """
    Create a heatmap figure for a labeled confusion matrix dataframe.

    Args:
        cm_df: Confusion matrix dataframe from ``labeled_confusion_matrix``.

    Returns:
        Matplotlib figure ready to display with ``st.pyplot``.
    """
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    return fig


def build_prediction_table(raw_df, probabilities, predictions):
    """
    Add model outputs to the input rows and arrange important columns first.

    Args:
        raw_df: Original dataset supplied for inference.
        probabilities: Survival probabilities produced by the model.
        predictions: Binary predictions produced from those probabilities.

    Returns:
        Dataframe containing the original rows plus probability, prediction,
        and readable prediction label columns.
    """
    results = raw_df.copy()
    results["Survival Probability"] = probabilities
    results["Predicted"] = predictions
    results["Predicted Label"] = results["Predicted"].map(LABEL_MAP)

    ordered_columns = [
        column
        for column in [
            "PassengerId",
            "Survived",
            "Predicted",
            "Predicted Label",
            "Survival Probability",
        ]
        if column in results.columns
    ]
    remaining_columns = [
        column for column in results.columns if column not in ordered_columns
    ]

    return results[ordered_columns + remaining_columns]


def run_inference(raw_df, model_path, preprocessor_path, threshold):
    """
    Run the full preprocessing and model scoring workflow for one dataframe.

    Args:
        raw_df: Input dataframe, optionally including a ``Survived`` label.
        model_path: Path to saved PyTorch model weights.
        preprocessor_path: Path to the fitted sklearn preprocessor.
        threshold: Decision threshold for converting probabilities to labels.

    Returns:
        Prediction table with original columns and model output columns.
    """
    has_labels = "Survived" in raw_df.columns
    X_raw = raw_df.drop(columns=["Survived"]) if has_labels else raw_df.copy()

    preprocessor = load_preprocessor(preprocessor_path)
    X_features = feature_engineering(X_raw)
    X_processed = preprocessor.transform(X_features)

    model = load_model(model_path, X_processed.shape[1])
    probabilities = predict_probabilities(model, X_processed)
    predictions = apply_threshold(probabilities, threshold)
    results = build_prediction_table(raw_df, probabilities, predictions)

    return results


def main():
    """
    Run the Streamlit inference application.

    The app loads user-selected dataset, model, and preprocessor paths, scores
    the dataset when requested, and displays predictions plus evaluation
    metrics when labels are available.
    """
    st.title("Titanic Inference")
    st.caption("Load a CSV and score it with the trained model")

    with st.sidebar:
        data_path = st.text_input("Dataset", "data/train.csv")
        model_path = st.text_input("Model weights", "model/model.pt")
        preprocessor_path = st.text_input("Preprocessor", "model/preprocessor.pkl")
        threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)
        run_button = st.button("Run inference", type="primary")

    paths = {
        "dataset": Path(data_path),
        "model": Path(model_path),
        "preprocessor": Path(preprocessor_path),
    }
    missing_paths = [name for name, path in paths.items() if not path.exists()]

    if missing_paths:
        st.error("Missing: " + ", ".join(missing_paths))
        st.stop()

    raw_df = load_dataset(paths["dataset"])
    st.subheader("Input Dataset")
    st.dataframe(raw_df.head(50), width="stretch")

    if not run_button:
        st.stop()

    try:
        results = run_inference(
            raw_df=raw_df,
            model_path=paths["model"],
            preprocessor_path=paths["preprocessor"],
            threshold=threshold,
        )
    except KeyError as exc:
        st.error(f"Dataset is missing a required column: {exc}")
        st.stop()
    except RuntimeError as exc:
        st.error("Model weights do not match the current preprocessing output.")
        st.exception(exc)
        st.stop()

    has_labels = "Survived" in results.columns
    summary = prediction_summary(results)

    st.subheader("Inference Metrics")
    show_prediction_summary(summary)

    tabs = st.tabs(["Results", "Evaluation"] if has_labels else ["Results"])

    with tabs[0]:
        st.subheader("Predictions")
        st.dataframe(results, width="stretch")

    if has_labels:
        with tabs[1]:
            y_true = results["Survived"].astype(int)
            y_pred = results["Predicted"].astype(int)
            metrics = metric_summary(y_true, y_pred)

            st.subheader("Evaluation Metrics")
            show_metrics(metrics)

            left_col, right_col = st.columns([1, 1])

            with left_col:
                st.subheader("Confusion Matrix")
                cm_df = labeled_confusion_matrix(y_true, y_pred)
                st.dataframe(cm_df, width="stretch")
                st.pyplot(plot_confusion_matrix(cm_df))

            with right_col:
                st.subheader("Classification Report")
                report = classification_report(
                    y_true,
                    y_pred,
                    labels=[0, 1],
                    target_names=CLASS_NAMES,
                    output_dict=True,
                    zero_division=0,
                )
                st.dataframe(pd.DataFrame(report).transpose(), width="stretch")


if __name__ == "__main__":
    st.set_page_config(page_title="Titanic Inference", layout="wide")
    main()
