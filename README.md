# Wearable Time Series Analysis for Fatigue Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF.svg)](https://www.kaggle.com/)

Predicting daily fatigue levels from passively collected wearable sensor data using time series feature extraction and machine learning.

![Results Summary](figures/fatigue_prediction_results.png)

---

## 🎯 Project Overview

This project demonstrates the use of **Fitbit wearable data** to predict self-reported **fatigue levels** through:

1. **Time series feature extraction** from minute-level activity and sleep data
2. **Temporal feature engineering** (lag features, rolling windows, circadian patterns)
3. **Cross-participant generalization testing** using Leave-One-Participant-Out CV
4. **Multi-model comparison** (Logistic Regression, Random Forest, XGBoost, Gradient Boosting)

### Relevance to Healthcare Applications

This work directly applies to longitudinal health monitoring scenarios:

| Application | Connection |
|-------------|------------|
| **Endometriosis monitoring** | Fatigue/pain prediction from wearables + PROMs |
| **Chronic disease management** | Remote patient symptom tracking |
| **Mental health assessment** | Depression/stress detection from behavioral patterns |

---

## 📊 Dataset

**PMData** — A sports and lifestyle logging dataset from Simula Research Laboratory.

| Attribute | Value |
|-----------|-------|
| Participants | 16 |
| Duration | 5 months (Nov 2019 – Mar 2020) |
| Wearable | Fitbit Versa 2 |
| Sensor data | Steps, heart rate, sleep, calories (minute-level) |
| Self-reports | Fatigue, mood, stress, sleep quality, soreness (daily) |

**Source:** [PMData on Hugging Face](https://huggingface.co/datasets/aai530-group6/pmdata) | [Original Paper](https://dl.acm.org/doi/10.1145/3339825.3394926)

---

## 🔧 Features Extracted

### Activity Features (from minute-level steps)
- `total_steps`, `active_minutes`, `sedentary_minutes`
- `steps_morning`, `steps_afternoon`, `steps_evening`, `steps_night`
- `peak_hour`, `std_steps`, `max_steps_min`

### Sleep Features (from Fitbit sleep logs)
- `sleep_hours`, `efficiency`, `minutesAsleep`, `minutesAwake`
- `timeInBed`, `minutesToFallAsleep`

### Temporal Features
- **Lag features:** Previous 1-2 days' activity and sleep
- **Rolling averages:** 3-day and 7-day windows
- **Calendar:** Day of week, weekend indicator

---

## 📈 Results

### Model Comparison (Leave-One-Participant-Out CV)

| Model | Weighted F1 | Low Fatigue F1 | High Fatigue F1 |
|-------|-------------|----------------|-----------------|
| **Logistic Regression** | **0.585** | **0.415** | 0.674 |
| Random Forest | 0.577 | 0.263 | 0.741 |
| XGBoost | 0.573 | 0.264 | 0.735 |
| Gradient Boosting | 0.566 | 0.296 | 0.708 |

### Key Findings

1. **Sleep features dominate:** `sleep_hours_roll7`, `minutesAsleep`, `timeInBed` are top predictors
2. **Circadian patterns matter:** Morning and nighttime activity correlate with fatigue state
3. **Large individual variation:** Per-participant F1 ranges from 0.28 to 0.88
4. **Temporal context helps:** Rolling averages capture cumulative sleep debt effects

### Per-Participant Generalization

![Per-participant F1](figures/cross_participant_f1.png)

The wide performance range (0.28–0.88) across participants highlights the need for **personalized models** in real-world deployment.

---

## 🚀 Quick Start

### Option 1: Run on Kaggle (Recommended)
[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/)

The notebook downloads the dataset automatically — no local setup required.

### Option 2: Run Locally

```bash
# Clone repository
git clone https://github.com/yourusername/wearable-fatigue-prediction.git
cd wearable-fatigue-prediction

# Create environment
conda create -n wearable python=3.10
conda activate wearable

# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook notebooks/wearable-fatigue-prediction.ipynb
```

---

## 📁 Repository Structure

```
wearable-fatigue-prediction/
├── README.md
├── requirements.txt
├── LICENSE
├── notebooks/
│   └── wearable-fatigue-prediction.ipynb
├── src/
│   ├── feature_extraction.py
│   └── data_loader.py
└── figures/
    ├── fatigue_prediction_results.png
    └── cross_participant_f1.png
```

---

## 📦 Requirements

```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
scikit-learn>=1.0.0
xgboost>=1.5.0
huggingface_hub>=0.10.0
```

---

## 📚 References

1. **PMData Dataset:**
   > Thambawita, V., et al. (2020). PMData: A Sports Logging Dataset. *Proceedings of the 11th ACM Multimedia Systems Conference*. https://doi.org/10.1145/3339825.3394926

2. **GLOBEM Benchmark:**
   > Xu, X., et al. (2022). GLOBEM: Cross-Dataset Generalization of Longitudinal Human Behavior Modeling. *IMWUT*. https://doi.org/10.1145/3569485

3. **Wearables for Health Monitoring:**
   > Jacobson, N.C., et al. (2021). Digital biomarkers of mood disorders and symptom change. *NPJ Digital Medicine*.

---

## 🔮 Future Work

- [ ] Add heart rate variability (HRV) features
- [ ] Implement personalized fine-tuning per participant
- [ ] Test on GLOBEM dataset for cross-dataset generalization
- [ ] Explore deep learning approaches (LSTM, Transformer)
- [ ] Add circadian rhythm analysis (cosinor fitting)

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Joana Owusu-Appiah**

---

## 🙏 Acknowledgments

- Simula Research Laboratory for the PMData dataset
- University of Washington for the GLOBEM benchmark platform
- Edinburgh CDT in Biomedical AI for project inspiration
