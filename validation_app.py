import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", ".matplotlib-cache")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from train import N_SPLITS, RANDOM_STATE


CLASS_NAMES = ["Not Survived", "Survived"]
LABEL_MAP = {0: "Not Survived", 1: "Survived"}
DATA_PATH = Path("data/train.csv")
CV_RESULTS_PATH = Path("model/cv_results.csv")
CV_PREDICTIONS_PATH = Path("model/cv_predictions.csv")
METADATA_PATH = Path("model/cv_metadata.json")


@st.cache_data
def load_data(data_path):
    return pd.read_csv(data_path)


@st.cache_data
def load_cv_results(path):
    return pd.read_csv(path)


@st.cache_data
def load_cv_predictions(path):
    predictions = pd.read_csv(path)
    predictions["fold"] = predictions["fold"].astype(int)
    predictions["row_index"] = predictions["row_index"].astype(int)
    predictions["actual"] = predictions["actual"].astype(int)
    return predictions


@st.cache_data
def load_metadata(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_threshold(cv_predictions, threshold):
    predictions = cv_predictions.copy()
    predictions["predicted"] = (
        predictions["survival_probability"] >= threshold
    ).astype(int)
    return predictions


def metric_summary(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }


def labeled_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return pd.DataFrame(
        cm,
        index=[f"True: {name}" for name in CLASS_NAMES],
        columns=[f"Predicted: {name}" for name in CLASS_NAMES],
    )


def show_metrics(metrics):
    cols = st.columns(4)
    for col, (name, value) in zip(cols, metrics.items()):
        col.metric(name, f"{value:.3f}")


def show_run_settings(metadata):
    settings = {
        "Best fold": metadata.get("best_fold", "Unknown"),
        "CV folds": metadata.get("n_splits", N_SPLITS),
        "Seed": metadata.get("random_state", RANDOM_STATE),
    }

    cols = st.columns(3)
    for col, (name, value) in zip(cols, settings.items()):
        col.metric(name, value)


def plot_confusion_matrix(cm_df):
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    return fig


def main():
    st.title("Titanic Validation Report")
    st.caption(f"Stratified {N_SPLITS}-fold cross-validation report")

    threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)

    paths = {
        "dataset": DATA_PATH,
        "cv results": CV_RESULTS_PATH,
        "cv predictions": CV_PREDICTIONS_PATH,
        "cv metadata": METADATA_PATH,
    }
    missing_paths = [name for name, path in paths.items() if not path.exists()]

    if missing_paths:
        st.error("Missing: " + ", ".join(missing_paths))
        st.stop()

    df = load_data(paths["dataset"])
    cv_results = load_cv_results(paths["cv results"])
    cv_predictions = load_cv_predictions(paths["cv predictions"])
    metadata = load_metadata(paths["cv metadata"])

    scored_predictions = apply_threshold(cv_predictions, threshold)
    overall_metrics = metric_summary(
        scored_predictions["actual"],
        scored_predictions["predicted"],
    )
    best_fold = int(metadata.get("best_fold", cv_results["fold"].iloc[0]))

    show_metrics(overall_metrics)
    show_run_settings(metadata)

    tabs = st.tabs(["Overview", "Folds", "Predictions"])

    with tabs[0]:
        left_col, right_col = st.columns([1, 1])

        with left_col:
            st.subheader("Out-of-Fold Confusion Matrix")
            cm_df = labeled_confusion_matrix(
                scored_predictions["actual"],
                scored_predictions["predicted"],
            )
            st.dataframe(cm_df, width="stretch")
            st.pyplot(plot_confusion_matrix(cm_df))

        with right_col:
            st.subheader("Fold Performance")
            chart_df = cv_results.set_index("fold")[["accuracy", "precision", "recall", "f1"]]
            st.line_chart(chart_df)

            summary_df = pd.DataFrame(
                {
                    "Mean": cv_results[["accuracy", "precision", "recall", "f1"]].mean(),
                    "Std": cv_results[["accuracy", "precision", "recall", "f1"]].std(),
                }
            )
            st.dataframe(summary_df, width="stretch")

        st.subheader("Cross-Validation Results")
        st.dataframe(cv_results, width="stretch")

    with tabs[1]:
        selected_fold = st.selectbox(
            "Fold",
            sorted(scored_predictions["fold"].unique()),
            index=best_fold - 1,
        )
        fold_predictions = scored_predictions[
            scored_predictions["fold"] == selected_fold
        ]
        fold_metrics = metric_summary(
            fold_predictions["actual"],
            fold_predictions["predicted"],
        )

        show_metrics(fold_metrics)

        fold_left, fold_right = st.columns([1, 1])

        with fold_left:
            st.subheader("Fold Confusion Matrix")
            fold_cm = labeled_confusion_matrix(
                fold_predictions["actual"],
                fold_predictions["predicted"],
            )
            st.pyplot(plot_confusion_matrix(fold_cm))

        with fold_right:
            st.subheader("Fold Classification Report")
            report = classification_report(
                fold_predictions["actual"],
                fold_predictions["predicted"],
                labels=[0, 1],
                target_names=CLASS_NAMES,
                output_dict=True,
                zero_division=0,
            )
            st.dataframe(pd.DataFrame(report).transpose(), width="stretch")

    with tabs[2]:
        selected_fold = st.selectbox(
            "Prediction fold",
            sorted(scored_predictions["fold"].unique()),
            index=best_fold - 1,
        )
        fold_predictions = scored_predictions[
            scored_predictions["fold"] == selected_fold
        ].copy()
        raw_rows = df.loc[fold_predictions["row_index"]].copy()
        raw_rows["Fold"] = fold_predictions["fold"].to_numpy()
        raw_rows["Actual"] = fold_predictions["actual"].map(LABEL_MAP).to_numpy()
        raw_rows["Predicted"] = fold_predictions["predicted"].map(LABEL_MAP).to_numpy()
        raw_rows["Survival Probability"] = fold_predictions[
            "survival_probability"
        ].to_numpy()

        st.dataframe(raw_rows, width="stretch")


if __name__ == "__main__":
    st.set_page_config(page_title="Titanic Validation Report", layout="wide")
    main()
