# COVID-19 Patient Risk Model Comparison

โปรเจกต์นี้เปรียบเทียบโมเดล Machine Learning จำนวน 2 โมเดล เพื่อทำนายสถานะการเสียชีวิตของผู้ป่วย COVID-19 จากข้อมูลพื้นฐานและโรคร่วมของผู้ป่วย

โปรเจกต์นี้จัดทำเพื่อการศึกษาและเปรียบเทียบโมเดลเท่านั้น ไม่ใช่เครื่องมือสำหรับวินิจฉัยหรือใช้ตัดสินใจทางการแพทย์

## Dataset

ไฟล์ข้อมูล:

`data/COVID19_Patient_Risk_Analysis.csv`

ข้อมูลมีทั้งหมด 25,000 แถว และ 38 คอลัมน์ โดยตัวแปรเป้าหมายมีสัดส่วนดังนี้:

| DEATH_STATUS | จำนวน | สัดส่วน |
|---|---:|---:|
| Survived | 23,166 | 92.66% |
| Died | 1,834 | 7.34% |

ข้อมูลมีความไม่สมดุลระหว่างคลาส จึงต้องใช้การแบ่งข้อมูลแบบ `stratify` และกำหนด `class_weight="balanced"` ให้โมเดลทั้งสองตัว

## Prediction task

ตัวแปรเป้าหมายคือ `DEATH_STATUS` ซึ่งถูกแปลงเป็น:

- `0` = Survived
- `1` = Died

โมเดลทั้งสองตัวจะใช้ feature, preprocessing, train/test split และ evaluation metrics ชุดเดียวกัน เพื่อให้การเปรียบเทียบมีความยุติธรรม

## Selected models

โปรเจกต์นี้เลือกเปรียบเทียบโมเดลต่อไปนี้:

### 1. Decision Tree

Decision Tree ค่อย ๆ เลือก feature ที่ช่วยแบ่งข้อมูลได้ดีที่สุดและสร้างกฎการตัดสินใจเป็นลำดับชั้น

เหตุผลที่เลือก:

- อธิบายกระบวนการตัดสินใจได้ง่าย
- แสดงความสำคัญของ feature ได้
- เรียนรู้ความสัมพันธ์ที่ไม่เป็นเส้นตรงได้
- ใช้เป็นโมเดลพื้นฐานที่เข้าใจง่ายสำหรับเปรียบเทียบ

ข้อควรระวังคือโมเดลอาจเกิด overfitting จึงต้องควบคุมค่าต่าง ๆ เช่น `max_depth`, `min_samples_split` และ `min_samples_leaf`

### 2. Support Vector Machine

Support Vector Machine หรือ SVM หาเส้นหรือขอบเขตที่แบ่งคลาสโดยให้ margin ระหว่างคลาสกว้างที่สุด

เหตุผลที่เลือก:

- เหมาะกับข้อมูลที่ผ่านการ scaling และ one-hot encoding
- สามารถสร้างขอบเขตการแบ่งคลาสแบบไม่เป็นเส้นตรงผ่าน kernel
- รองรับข้อมูลคลาสไม่สมดุลด้วย `class_weight="balanced"`
- มีแนวคิดต่างจาก Decision Tree อย่างชัดเจน จึงเหมาะสำหรับการเปรียบเทียบ

ข้อควรระวังคือ SVM ใช้เวลาเทรนมากกว่า Decision Tree และต้องปรับค่า `C`, `kernel` และ `gamma` อย่างเหมาะสม

## Why the other models were not selected

- K-Nearest Neighbors ใช้ระยะห่างระหว่างตัวอย่าง การมี feature จาก one-hot encoding จำนวนมากอาจทำให้ระยะห่างมีความหมายน้อยลง และการทำนายจะช้าลงเมื่อข้อมูลมีจำนวนมาก
- Naive Bayes สมมติว่า features ค่อนข้างเป็นอิสระต่อกัน แต่ข้อมูลโรคร่วม เช่น เบาหวาน ความดันโลหิตสูง โรคหัวใจ และโรคอ้วน อาจมีความสัมพันธ์กัน

โมเดลที่ไม่ได้เลือกไม่ได้หมายความว่าเป็นโมเดลที่ไม่ดี แต่ Decision Tree และ SVM เหมาะกับวัตถุประสงค์การเปรียบเทียบของโปรเจกต์นี้มากกว่า

## Features and data leakage

โมเดลใช้ข้อมูลพื้นฐานและโรคร่วมที่กำหนดไว้ใน `src/preprocess.py` เช่น:

- อายุ
- เพศ
- ประเภทผู้ป่วย
- ปอดอักเสบ
- เบาหวาน
- COPD
- หอบหืด
- ภูมิคุ้มกันบกพร่อง
- ความดันโลหิตสูง
- โรคหัวใจและหลอดเลือด
- โรคอ้วน
- โรคไตเรื้อรัง
- การสูบบุหรี่

คอลัมน์ที่ทราบผลลัพธ์หลังเกิดเหตุการณ์จะไม่ถูกใช้เป็น feature เช่น:

- `DATE_DIED`
- `DEATH_YEAR`
- `DEATH_MONTH`
- `RECOVERY_STATUS`
- `RISK_CATEGORY`

การตัดคอลัมน์เหล่านี้ช่วยลดความเสี่ยงของ data leakage

## Shared preprocessing

ไฟล์ส่วนกลางที่โมเดลทั้งสองตัวต้องใช้ร่วมกัน:

- `src/preprocess.py` อ่านข้อมูล เลือก features จัดการ missing values และแบ่ง train/test
- `src/evaluate.py` คำนวณ metrics และบันทึกผลการทดลอง
- `requirements.txt` ระบุไลบรารีที่ใช้ในโปรเจกต์

ค่ากลางที่ใช้ร่วมกัน:

- `test_size=0.20`
- `random_state=42`
- `stratify=y`
- Numeric missing values ใช้ค่ามัธยฐาน
- Categorical missing values ใช้ค่าที่พบบ่อยที่สุด
- Categorical features ใช้ One-Hot Encoding
- Numeric features ใช้ Standard Scaling

Decision Tree ไม่จำเป็นต้องใช้ scaling แต่จะใช้ preprocessing ชุดเดียวกับ SVM เพื่อควบคุมเงื่อนไขการทดลองให้เหมือนกัน

## Evaluation metrics

ข้อมูลกลุ่ม `Died` มีเพียง 7.34% จึงไม่ใช้ Accuracy เป็นเกณฑ์ตัดสินเพียงอย่างเดียว

Metrics ที่ใช้:

- Accuracy
- Precision ของกลุ่ม `Died`
- Recall ของกลุ่ม `Died`
- F1-score ของกลุ่ม `Died`
- ROC-AUC
- PR-AUC
- Confusion Matrix

เกณฑ์หลักในการเลือกโมเดลคือ F1-score ของกลุ่ม `Died` หากผลใกล้เคียงกันจะพิจารณา PR-AUC และ Recall ประกอบ

ROC-AUC และ PR-AUC จะคำนวณจากค่าความน่าจะเป็นหรือ continuous decision score ของโมเดล ไม่ใช่จาก class prediction เพียงอย่างเดียว

## Expected result comparison

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Decision Tree | รอทดลอง | รอทดลอง | รอทดลอง | รอทดลอง | รอทดลอง | รอทดลอง |
| SVM | รอทดลอง | รอทดลอง | รอทดลอง | รอทดลอง | รอทดลอง | รอทดลอง |

ยังไม่กำหนดล่วงหน้าว่าโมเดลใดดีที่สุด ผลสรุปจะมาจากการทดลองบน test set ชุดเดียวกัน

## Project structure

```text
kaggle-model/
├── data/
│   └── COVID19_Patient_Risk_Analysis.csv
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── evaluate.py
│   └── models/
│       ├── __init__.py
│       ├── decision_tree.py
│       └── svm.py
├── results/
│   ├── decision_tree_metrics.json
│   └── svm_metrics.json
├── .gitignore
├── README.md
└── requirements.txt
```

ไฟล์โมเดลและไฟล์ผลลัพธ์ในโครงสร้างข้างต้นจะถูกสร้างเมื่อเริ่มพัฒนาแต่ละโมเดล

## Project members

- สมาชิกคนที่ 1 พัฒนาและทดสอบ Decision Tree บน branch `model-1`
- สมาชิกคนที่ 2 พัฒนาและทดสอบ SVM บน branch `model-2`
- สมาชิกทั้งสองคนตรวจสอบ Pull Request และสรุปผลร่วมกัน

## Git workflow

`main` เก็บเฉพาะงานที่ผ่านการตรวจสอบแล้ว สมาชิกแต่ละคนต้องทำงานบน branch ของตนเองและเปิด Pull Request ก่อน merge

ก่อนเริ่มงานทุกครั้ง:

```powershell
git switch main
git pull origin main
git switch ชื่อ-branch-ของตนเอง
git merge main
```

หลังแก้ไขโค้ด:

```powershell
git status
git add ชื่อไฟล์
git commit -m "อธิบายสิ่งที่เปลี่ยน"
git push
```

ห้าม push โค้ดโมเดลเข้า `main` โดยตรง

## Installation

สร้าง virtual environment:

```powershell
python -m venv .venv
```

เปิดใช้งาน virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

ติดตั้งไลบรารี:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

หาก PowerShell ไม่อนุญาตให้เปิด virtual environment ให้กำหนดสิทธิ์เฉพาะหน้าต่างปัจจุบัน:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Test shared preprocessing

ตั้งค่า UTF-8 สำหรับหน้าต่าง PowerShell ปัจจุบัน:

```powershell
$env:PYTHONUTF8="1"
```

ทดสอบ preprocessing:

```powershell
python -m src.preprocess
```

ผลที่คาดว่าจะได้รับ:

- จำนวนข้อมูลทั้งหมด 25,000 แถว
- จำนวน features 18 ตัว
- จำนวนข้อมูล train 20,000 แถว
- จำนวนข้อมูล test 5,000 แถว
- สัดส่วน `Died` ใน train และ test ประมาณ 7.34%

## Run models

หลังจากพัฒนาโมเดลเสร็จแล้ว จะรันด้วยคำสั่ง:

```powershell
python -m src.models.decision_tree
python -m src.models.svm
```

ผลการประเมินจะถูกบันทึกไว้ในโฟลเดอร์ `results/` เพื่อใช้สร้างตารางเปรียบเทียบและสรุปผล

## Current status

- [x] เพิ่มชุดข้อมูล
- [x] สร้าง preprocessing กลาง
- [x] สร้าง evaluation กลาง
- [x] กำหนด target และ evaluation metrics
- [x] เลือก Decision Tree และ SVM
- [ ] พัฒนา Decision Tree
- [ ] พัฒนา SVM
- [ ] เปรียบเทียบผลการทดลอง
- [ ] สรุปโมเดลที่มีประสิทธิภาพดีกว่า
