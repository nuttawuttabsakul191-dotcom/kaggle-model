from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.evaluate import evaluate_classifier, print_metrics, save_metrics
from src.preprocess import build_preprocessor, get_train_test_data

# กำหนด Path สำหรับบันทึกผลลัพธ์
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "results" / "svm_metrics.json"


def train_and_evaluate():
    # 1. โหลดข้อมูล train/test
    print("กำลังโหลดและแบ่งข้อมูล train/test...")
    X_train, X_test, y_train, y_test = get_train_test_data()

    # 2. สร้าง Pipeline ด้วย build_preprocessor และ SVC
    print("กำลังสร้างและฝึกสอนโมเดล SVM...")
    preprocessor = build_preprocessor()
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    # 3. ฝึกสอนโมเดล
    model.fit(X_train, y_train)

    # 4. ทำนายผล
    print("กำลังทำนายผลบนชุดทดสอบ...")
    y_pred = model.predict(X_test)
    y_score = model.decision_function(X_test)

    # 5. ประเมินผล
    metrics = evaluate_classifier(
        y_true=y_test,
        y_pred=y_pred,
        y_score=y_score,
    )

    # 6. แสดงผลลัพธ์
    print_metrics(metrics)

    # 7. บันทึกผลลัพธ์ลงไฟล์
    save_metrics(metrics, OUTPUT_PATH)

    return model, metrics


if __name__ == "__main__":
    train_and_evaluate()
