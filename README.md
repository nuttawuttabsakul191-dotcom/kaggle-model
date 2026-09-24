# COVID-19 Patient Risk Model Comparison

โปรเจกต์นี้เปรียบเทียบประสิทธิภาพของโมเดล Machine Learning จำนวน 2 โมเดล เพื่อทำนายสถานะการเสียชีวิตของผู้ป่วย COVID-19 จากข้อมูลพื้นฐานและโรคร่วมของผู้ป่วย

> โปรเจกต์นี้จัดทำเพื่อการศึกษาเท่านั้น ไม่ใช่เครื่องมือสำหรับวินิจฉัยหรือใช้ตัดสินใจทางการแพทย์

## เป้าหมายของการทำนาย

ตัวแปรเป้าหมายคือ `DEATH_STATUS`

- `0` = Survived
- `1` = Died

โมเดลที่นำมาเปรียบเทียบ:

1. Decision Tree
2. Support Vector Machine (SVM)

## Dataset

ไฟล์ข้อมูลอยู่ที่:

```text
data/COVID19_Patient_Risk_Analysis.csv
```

ข้อมูลมีทั้งหมด 25,000 แถว และ 38 คอลัมน์ โดยมีสัดส่วนของตัวแปรเป้าหมายดังนี้:

| DEATH_STATUS | จำนวน | สัดส่วน |
|---|---:|---:|
| Survived | 23,166 | 92.66% |
| Died | 1,834 | 7.34% |

ข้อมูลมีความไม่สมดุลระหว่างคลาส จึงไม่ใช้ Accuracy เป็นเกณฑ์ตัดสินเพียงอย่างเดียว

## สมาชิกและหน้าที่

| สมาชิก | GitHub | Branch | งานที่รับผิดชอบ |
|---|---|---|---|
| สมาชิกคนที่ 1 | `@nuttawuttabsakul191-dotcom` | `model-1` | Decision Tree และ `notebooks/decision_tree.ipynb` |
| สมาชิกคนที่ 2 | `@panrawoot1-dot` | `model-2` | SVM และ `notebooks/svm.ipynb` |

สมาชิกแต่ละคนพัฒนาโมเดลใน Notebook ของตนเอง และเปิด Pull Request เข้า `main` เพื่อให้อีกคนตรวจสอบก่อน merge

## Project structure

```text
kaggle-model/
├── data/
│   └── COVID19_Patient_Risk_Analysis.csv
├── notebooks/
│   ├── decision_tree.ipynb
│   └── svm.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   └── evaluate.py
├── results/
│   ├── decision_tree_metrics.json
│   └── svm_metrics.json
├── .gitignore
├── README.md
└── requirements.txt
```

ไฟล์ใน `src/` เป็นโค้ดส่วนกลาง ส่วนไฟล์ใน `notebooks/` เป็นพื้นที่พัฒนา ทดลอง และอธิบายผลของแต่ละโมเดล

## กติกาการเปรียบเทียบโมเดล

เพื่อให้เปรียบเทียบอย่างยุติธรรม ทั้งสอง Notebook ต้องใช้:

- Feature ชุดเดียวกันจาก `src/preprocess.py`
- Train/test split ชุดเดียวกัน
- `test_size=0.20`
- `random_state=42`
- `stratify=y`
- `class_weight="balanced"`
- ฟังก์ชันประเมินผลชุดเดียวกันจาก `src/evaluate.py`

คอลัมน์ที่ทราบผลลัพธ์หลังเกิดเหตุการณ์ เช่น `DATE_DIED`, `DEATH_YEAR`, `DEATH_MONTH`, `RECOVERY_STATUS` และ `RISK_CATEGORY` จะไม่ถูกนำมาใช้เป็น feature เพื่อป้องกัน data leakage

## โครงสร้างที่ต้องมีใน Notebook

Notebook ของทั้งสองโมเดลควรเรียงหัวข้อตามลำดับเดียวกัน:

1. อธิบายวัตถุประสงค์ของโมเดล
2. Import libraries และไฟล์ส่วนกลางจาก `src/`
3. โหลดและแบ่งข้อมูลด้วย `src/preprocess.py`
4. แสดงจำนวนและสัดส่วนของแต่ละคลาส
5. สร้างโมเดลและกำหนดพารามิเตอร์
6. เทรนโมเดลด้วย training set
7. ทำนายผลด้วย test set
8. ประเมินผลด้วย `src/evaluate.py`
9. แสดง Confusion Matrix และผลลัพธ์ที่สำคัญ
10. บันทึก metrics เป็นไฟล์ JSON ใน `results/`
11. เขียนสรุปผลของโมเดล

Notebook ต้องใช้ test set สำหรับประเมินผลสุดท้ายเท่านั้น หากมีการปรับพารามิเตอร์ให้ใช้ข้อมูล training set หรือ cross-validation เพื่อไม่ให้เกิด data leakage

## Evaluation metrics

Metrics ที่ใช้เปรียบเทียบ:

- Accuracy
- Precision ของคลาส `Died`
- Recall ของคลาส `Died`
- F1-score ของคลาส `Died`
- ROC-AUC
- PR-AUC
- Confusion Matrix

เกณฑ์หลักในการเลือกโมเดลคือ F1-score ของคลาส `Died` และพิจารณา PR-AUC กับ Recall ประกอบ เนื่องจากข้อมูลคลาส `Died` มีจำนวนน้อยกว่าคลาส `Survived` มาก

## Installation

สร้างและเปิดใช้งาน virtual environment ใน PowerShell:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

ติดตั้งไลบรารี:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

เพิ่ม virtual environment เป็น Jupyter kernel:

```powershell
python -m ipykernel install --user --name kaggle-model --display-name "Python (kaggle-model)"
```

## การเปิด Notebook ใน VS Code

ติดตั้งส่วนขยาย **Python** และ **Jupyter** ใน VS Code จากนั้นเปิด Notebook:

```powershell
code notebooks/decision_tree.ipynb
```

หรือ:

```powershell
code notebooks/svm.ipynb
```

ที่มุมขวาบนของ Notebook ให้กด **Select Kernel** แล้วเลือก `Python (kaggle-model)` จากนั้นกด **Run All**

## Git workflow

ก่อนเริ่มงานทุกครั้ง ให้นำ `main` ล่าสุดมารวมกับ branch ของตนเอง:

```powershell
git switch main
git pull origin main
git switch ชื่อ-branch-ของตนเอง
git merge main
```

หลังแก้ไข Notebook:

```powershell
git status
git add notebooks/ชื่อไฟล์.ipynb results/ชื่อไฟล์_metrics.json
git commit -m "อธิบายงานที่ทำ"
git push origin ชื่อ-branch-ของตนเอง
```

จากนั้นเปิด Pull Request โดยกำหนด:

```text
base: main
compare: branch ของผู้พัฒนา
```

ห้าม push งานเข้า `main` โดยตรง เพื่อให้ GitHub แสดงประวัติการทำงานและการตรวจสอบของสมาชิกแต่ละคนอย่างชัดเจน

## ตารางสรุปผล

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Decision Tree | รอสรุปผล | รอสรุปผล | รอสรุปผล | รอสรุปผล | รอสรุปผล | รอสรุปผล |
| SVM | รอสรุปผล | รอสรุปผล | รอสรุปผล | รอสรุปผล | รอสรุปผล | รอสรุปผล |

ผลสรุปจะมาจาก test set ชุดเดียวกัน หลังจาก Notebook ของทั้งสองโมเดลถูกรันด้วยเงื่อนไขเดียวกัน

## สถานะโปรเจกต์

- [x] เพิ่ม Dataset
- [x] สร้าง preprocessing ส่วนกลาง
- [x] สร้าง evaluation ส่วนกลาง
- [x] กำหนดโมเดลและ metrics ที่ใช้เปรียบเทียบ
- [x] สร้าง Decision Tree Notebook
- [ ] สร้าง SVM Notebook จากโค้ด SVM ที่พัฒนาไว้
- [ ] ตรวจสอบและรัน Notebook ทั้งสองไฟล์
- [ ] เปรียบเทียบผลลัพธ์ของสองโมเดล
- [ ] สรุปว่าโมเดลใดเหมาะสมกว่า
