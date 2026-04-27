# ⚙️ CNC Tool Wear Detection System

## 📌 Overview
An ML-based tool condition monitoring system using real CNC sensor 
data from the University of Michigan SMART Lab. 
This project simulates a real Industry 4.0 predictive maintenance 
system used in smart manufacturing.

## 🏭 Problem Statement
Tool wear in CNC machining causes poor surface finish, dimensional 
errors and machine damage. Manual inspection is costly and causes 
downtime. This system automatically detects worn tools from 
motor sensor signals — no manual inspection needed.

## 📊 Dataset
- **Source:** University of Michigan SMART Lab (Kaggle)
- **Size:** 18 CNC milling experiments
- **Signals:** X, Y, Z axis + Spindle motor (velocity, 
  acceleration, current, power)
- **Target:** Binary classification — worn vs unworn tool

## ⚙️ Methodology
1. **Feature Engineering** — Extracted mean, RMS, std, max 
   from 12 sensor signals across active cutting stages only
2. **Model Comparison** — Tested KNN, SVM, Random Forest, 
   Decision Tree
3. **Validation** — Leave-One-Out Cross Validation 
   (standard for small datasets)
4. **Result** — KNN (K=3) achieved 67% accuracy with 
   **90% recall on worn tools**

## 🖥️ Dashboard
Interactive Streamlit dashboard for real-time prediction:
- Upload any experiment CSV
- Instantly predicts tool condition
- Shows live sensor signal plots

## 🛠️ Tools & Technologies
- Python, Scikit-learn, Pandas, NumPy
- Streamlit, Matplotlib, Seaborn
- Google Colab, GitHub

## 🚀 How to Run
```bash
pip install streamlit scikit-learn pandas numpy matplotlib seaborn joblib
streamlit run app.py
```

## 📈 Results
| Model | Accuracy |
|---|---|
| KNN (K=3) | 66.7% |
| Random Forest | 44.4% |
| Decision Tree | 38.9% |
| SVM | 33.3% |

**Worn tool recall: 90%** — critical metric for manufacturing

## 👤 Author
Binet N B — M.Tech Computer Integrated Manufacturing  
National Institute of Technology, Warangal
