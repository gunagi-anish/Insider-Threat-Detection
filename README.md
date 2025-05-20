# Insider Threat Detection using Learning Adaptive Neighbors (LAN)

This project implements an insider threat detection system using the Learning Adaptive Neighbors (LAN) algorithm on the CERT R4.2 or R5.2 datasets.

---

## 📁 Project Structure

```
insider-threat-lan/
├── data/
│   ├── raw/                 # Raw CERT logs (logon.csv, file.csv, etc.)
│   └── processed/           # Generated features.csv
├── models/
│   └── lan_model.py         # LAN model implementation
├── preprocessing/
│   ├── extract_features.py  # Feature engineering from logs
│   └── label_users.py       # Label users as malicious or benign
├── evaluation/
│   └── evaluate.py          # ROC and PR evaluation
├── scripts/
│   └── generate_features_csv.py  # Script to create features.csv
├── main.py                  # Main pipeline script
└── requirements.txt         # Python dependencies
```

---

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/insider-threat-lan.git
cd insider-threat-lan
```

### 2. Set Up Python Environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Prepare the Dataset

* Download the CERT R4.2 or R5.2 dataset from [SEI CERT Website](https://resources.sei.cmu.edu/library/asset-view.cfm?assetid=508099).
* Place the extracted raw log files (logon.csv, file.csv, email.csv, etc.) in `data/raw/`.

### 4. Generate Features

```bash
python scripts/generate_features_csv.py
```

This creates the `features.csv` file in `data/processed/`.

### 5. Run the Detection Pipeline

```bash
python main.py
```

This trains the LAN model and prints ROC and PR AUC scores, along with their plots.

---

## 📊 Output Example

```
ROC AUC: 0.83
PR AUC: 0.56
```

And two plots:

* Precision-Recall Curve
* ROC Curve

---

## 🧪 Optional: Use Jupyter Notebook

```bash
pip install notebook
jupyter notebook
```

---

## 📌 Dependencies

See `requirements.txt`:

* numpy
* pandas
* scikit-learn
* matplotlib
* seaborn

---

## 📬 Contact

For questions or suggestions, open an issue or contact the project maintainer.

---

## ✅ License

This project is for academic and research use only.
