from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# หาโฟลเดอร์หลักของโปรเจกต์โดยอัตโนมัติ
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "COVID19_Patient_Risk_Analysis.csv"
)

TARGET_COLUMN = "DEATH_STATUS"

# AGE เป็นข้อมูลตัวเลขจริง
NUMERIC_FEATURES = [
    "AGE",
]

# คอลัมน์เหล่านี้เป็นรหัสกลุ่ม ไม่ควรถือว่าเป็นตัวเลขต่อเนื่อง
CATEGORICAL_FEATURES = [
    "USMER",
    "MEDICAL_UNIT",
    "SEX",
    "PATIENT_TYPE",
    "PNEUMONIA",
    "PREGNANT",
    "DIABETES",
    "COPD",
    "ASTHMA",
    "INMSUPR",
    "HIPERTENSION",
    "OTHER_DISEASE",
    "CARDIOVASCULAR",
    "OBESITY",
    "RENAL_CHRONIC",
    "TOBACCO",
    "CLASIFFICATION_FINAL",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_dataset(data_path=DATA_PATH):
    """
    อ่าน CSV และคืนค่า X กับ y

    y = 0 หมายถึง Survived
    y = 1 หมายถึง Died
    """
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"ไม่พบไฟล์ข้อมูล: {data_path}"
        )

    df = pd.read_csv(data_path, low_memory=False)

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"ไม่พบคอลัมน์ที่จำเป็น: {missing_columns}"
        )

    X = df[FEATURE_COLUMNS].copy()

    # แปลง feature ให้เป็นตัวเลข
    for column in FEATURE_COLUMNS:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    # ในชุดข้อมูลนี้ 97, 98 และ 99 มักหมายถึง
    # ไม่ทราบข้อมูลหรือไม่เกี่ยวข้อง
    # ไม่เปลี่ยนค่า AGE เพราะอายุ 97-99 ปีอาจเป็นค่าจริง
    for column in CATEGORICAL_FEATURES:
        X[column] = X[column].replace(
            [97, 98, 99],
            np.nan,
        )

    y = df[TARGET_COLUMN].map(
        {
            "Survived": 0,
            "Died": 1,
        }
    )

    # เก็บเฉพาะแถวที่ target ถูกต้อง
    valid_target = y.notna()

    X = X.loc[valid_target].reset_index(drop=True)
    y = y.loc[valid_target].astype(int).reset_index(drop=True)

    return X, y


def split_dataset(
    X,
    y,
    test_size=0.20,
    random_state=42,
):
    """
    แบ่งข้อมูล train/test

    stratify=y ทำให้สัดส่วน Died และ Survived
    ใกล้เคียงกันทั้งชุด train และ test
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def build_preprocessor():
    """
    สร้าง preprocessing ที่นำไปใช้ร่วมกันได้ทุกโมเดล
    """
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return preprocessor


def get_train_test_data():
    """
    ฟังก์ชันหลักที่โมเดลทั้งสองตัวจะเรียกใช้
    """
    X, y = load_dataset()

    return split_dataset(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )


if __name__ == "__main__":
    X, y = load_dataset()

    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y,
    )

    print(f"จำนวนข้อมูลทั้งหมด: {len(X):,}")
    print(f"จำนวน features: {X.shape[1]}")
    print(f"จำนวนข้อมูล train: {len(X_train):,}")
    print(f"จำนวนข้อมูล test: {len(X_test):,}")

    print("\nจำนวน target ทั้งหมด:")
    print(y.value_counts().rename(index={0: "Survived", 1: "Died"}))

    print("\nสัดส่วน target ในชุด train:")
    print(
        y_train.value_counts(normalize=True)
        .rename(index={0: "Survived", 1: "Died"})
        .round(4)
    )

    print("\nสัดส่วน target ในชุด test:")
    print(
        y_test.value_counts(normalize=True)
        .rename(index={0: "Survived", 1: "Died"})
        .round(4)
    )