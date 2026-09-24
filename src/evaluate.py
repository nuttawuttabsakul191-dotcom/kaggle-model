import json
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_classifier(
    y_true,
    y_pred,
    y_score=None,
):
    """
    ประเมินผลโมเดล Binary Classification

    y_true  = ค่าจริง
    y_pred  = class ที่โมเดลทำนาย
    y_score = ความน่าจะเป็นของ class Died
    """
    metrics = {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Survived", "Died"],
            zero_division=0,
            output_dict=True,
        ),
    }

    if y_score is not None:
        metrics["roc_auc"] = float(
            roc_auc_score(y_true, y_score)
        )
        metrics["pr_auc"] = float(
            average_precision_score(y_true, y_score)
        )

    return metrics


def print_metrics(metrics):
    """
    แสดงผล metric สำคัญบนหน้าจอ
    """
    print("\nผลการประเมินโมเดล")
    print("-" * 30)

    metric_names = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "pr_auc",
    ]

    for name in metric_names:
        if name in metrics:
            print(f"{name}: {metrics[name]:.4f}")

    matrix = metrics["confusion_matrix"]

    print("\nConfusion Matrix")
    print("                 ทำนาย Survived  ทำนาย Died")
    print(f"จริง Survived       {matrix[0][0]:>8}  {matrix[0][1]:>11}")
    print(f"จริง Died           {matrix[1][0]:>8}  {matrix[1][1]:>11}")


def save_metrics(metrics, output_path):
    """
    บันทึกผลเป็น JSON เพื่อใช้เปรียบเทียบภายหลัง
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print(f"\nบันทึกผลไว้ที่: {output_path}")