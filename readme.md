# Irish Public Transport Delay Prediction  

This project focused on machine learning and deep learning models for predicting Irish public transport delays using GTFS real-time data. This project implements 12 classification models to predict service delays with up to 93.53% accuracy.

## Table of Contents
- [Project Overview](#project-overview)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Key Results](#key-results)
- [Author](#author)

---

## Project Overview

This repository contains the machine learning component of an Irish public transport delay analysis project. Using real-time GTFS data from Transport for Ireland, this work develops and evaluates multiple classification models to predict whether a transport service will be delayed.

### Business Problem

Transport for Ireland (TFI) faces critical challenges in providing reliable and punctual service across its multi-modal transport network. Service delays impact passenger satisfaction, operational efficiency and public transport adoption rates. This project addresses:

1. **Delay Prediction**: Binary classification to predict if a service will be delayed (is_delayed: 1=delayed, 0=on-time/early)
2. **Model Comparison**: Evaluation of traditional ML vs. deep learning approaches
3. **Explainability**: LIME analysis to understand feature importance and model decisions
4. **Operational Insights**: Data-driven recommendations for transport authorities

### Project Objectives

- Develop accurate predictive models for transport service delays
- Compare traditional machine learning against deep learning architectures
- Identify key operational factors driving delays
- Provide interpretable insights for stakeholders using explainable AI

---

## Model Performance

### Performance Summary

| Rank | Model | Accuracy | Type | Training Time | Status |
|------|-------|----------|------|---------------|--------|
| 1 | **Random Forest** | **93.53%** | Traditional ML | ~2 seconds | **Recommended** |
| 2 | Simple Neural Network | 90.37% | Deep Learning | ~45 seconds | High Performance |
| 3 | Decision Tree | 89.37% | Traditional ML | <1 second | Good |
| 4 | K-Nearest Neighbors | 88.39% | Traditional ML | ~1 second | Good |
| 5 | Transformer | 86.18% | Deep Learning | ~90 seconds | Acceptable |
| 6 | Simple RNN | 79.53% | Deep Learning | ~60 seconds | Moderate |
| 7 | SVM | 78.67% | Traditional ML | ~3 seconds | Moderate |
| 8 | Logistic Regression | 73.30% | Traditional ML | <1 second | Baseline |
| 9 | Naive Bayes | 68.96% | Traditional ML | <1 second | Baseline |
| 10 | BiLSTM | 69.19% | Deep Learning | ~120 seconds | Poor |
| 11 | GRU | 69.16% | Deep Learning | ~75 seconds | Poor |
| 12 | LSTM | 66.37% | Deep Learning | ~80 seconds | Poor |

### Recommended Model: Random Forest

**Random Forest is the best choice for production deployment** due to:
- Highest accuracy (93.53%)
- Fast training and inference (<2 seconds)
- Built-in feature importance analysis
- Robust to overfitting with ensemble approach
- No hyperparameter tuning required for strong baseline performance

### Why Deep Learning Underperformed

Deep learning models (LSTM, GRU, BiLSTM) showed poor performance because:
- **Small dataset size**: 16,169 observations insufficient for complex neural networks
- **Tabular data structure**: No sequential or temporal dependencies to leverage
- **Feature complexity**: Simple numeric/categorical features don't benefit from deep architectures
- **Computational overhead**: Training time 40-60x longer with worse accuracy

---


## Dataset

### Dataset Overview

**Source**: Transport for Ireland GTFS Realtime API  
**Collection Date**: December 16, 2024  
**Collection Time**: 2:34 PM - 2:45 PM  
**Total Observations**: 16169 records  
**Features**: 15 variables  

### Key Features

| Feature Category | Variables | Description |
|-----------------|-----------|-------------|
| **Target Variable** | `is_delayed` | Binary (1=delayed, 0=on-time/early) |
| **Delay Metrics** | `arrival_delay`, `departure_delay` | Delay in seconds (negative=early, positive=late) |
| **Geographic** | `latitude`, `longitude`, `stop_lat`, `stop_lon` | GPS coordinates |
| **Route Information** | `route_id`, `route_short_name`, `agency_name` | Service identifiers |
| **Temporal** | `timestamp`, `hour`, `day_of_week` | Time-based features |
| **Operational** | `trip_id`, `stop_id` | Trip and stop identifiers |

### Data Preprocessing

1. **Missing Value Treatment**: Imputation for numeric features, mode for categorical
2. **Feature Engineering**: Created `hour`, `day_of_week` from timestamps
3. **Encoding**: 
   - OneHotEncoding for categorical variables (route_id, agency_name)
   - Label encoding for ordinal features
4. **Scaling**: StandardScaler applied to all numeric features
5. **Train-Test Split**: 80% training, 20% testing (stratified)

### Data Source Note

This dataset was prepared through a comprehensive data mining pipeline documented in a separate repository. The pipeline includes:
- GTFS Realtime API authentication and data collection
- Protocol Buffer parsing
- Merging real-time data with static GTFS reference files
- Feature engineering and data quality validation

For full data collection methodology, see the companion repository: [Irish Public Transport Delay Data Mining Analysis](https://github.com/wendytnn/irish-public-transport-delay-data-mining-analysis)

---

## Methodology

### Machine Learning Pipeline

This project follows the **CRISP-DM (Cross-Industry Standard Process for Data Mining)** methodology:

#### 1. Business Understanding
- Define transport reliability challenges
- Identify stakeholders (TFI, passengers, operators)
- Establish success metrics (accuracy, precision, recall, F1-score)

#### 2. Data Understanding
- Exploratory Data Analysis (EDA)
- Distribution analysis of delays
- Correlation analysis between features
- Class imbalance assessment

#### 3. Data Preparation
- Missing value handling
- Feature engineering (temporal features)
- Encoding categorical variables
- Feature scaling and normalization
- Train-test split

#### 4. Modeling

**Traditional Machine Learning (6 models):**
1. Logistic Regression - Linear baseline
2. Decision Tree - Non-linear baseline
3. Random Forest - Ensemble method
4. Support Vector Machine (SVM) - Kernel-based
5. K-Nearest Neighbors (KNN) - Instance-based
6. Naive Bayes - Probabilistic

**Deep Learning (6 models):**
1. Simple Neural Network - Feedforward baseline
2. LSTM - Long Short-Term Memory
3. GRU - Gated Recurrent Unit
4. BiLSTM - Bidirectional LSTM
5. Simple RNN - Basic recurrent network
6. Transformer - Attention-based architecture

#### 5. Evaluation

**Metrics:**
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC-AUC Score

**Validation:**
- Train-test split (80/20)
- Cross-validation for model selection
- Holdout test set for final evaluation

#### 6. Explainability

**LIME (Local Interpretable Model-agnostic Explanations):**
- Instance-level feature contribution analysis
- Model decision transparency
- Stakeholder-friendly interpretations

---

## Technologies Used

### Core Libraries

**Machine Learning:**
- `scikit-learn 1.3+` - Traditional ML algorithms, preprocessing, metrics
- `tensorflow 2.13+` / `keras` - Deep learning models
- `lime 0.2+` - Explainable AI

**Data Processing:**
- `pandas 2.0+` - Data manipulation
- `numpy 1.24+` - Numerical computing

**Visualization:**
- `matplotlib 3.7+` - Static plots
- `seaborn 0.12+` - Statistical visualizations

### Development Environment
- **Python 3.8+**
- **Jupyter Notebook** - Interactive development
- **Git** - Version control

--- 

## Author

**Tan Pei Wen (Wendy)**  
*Final Year Mathematics and Data Science Student*  
Dundalk Institute of Technology (DkIT), Ireland

**Student ID**: D00253240  
**Project Date**: December 2024  
**Academic Year**: 2024/2025

**Related Work**:
- Data Mining Analysis: [Irish Public Transport Delay Data Mining](https://github.com/wendytnn/irish-public-transport-delay-data-mining-analysis)   

--- 

## Acknowledgments

- **Transport for Ireland (TFI)** and **National Transport Authority (NTA)** for open GTFS API access
- **Dundalk Institute of Technology** for academic supervision and resources
- **PolicyStreet** for industry experience informing practical ML applications
- **scikit-learn** and **TensorFlow** communities for excellent documentation

---

## References

1. Transport for Ireland. (2024). *GTFS Realtime API Documentation*. https://developer.nationaltransport.ie/
2. Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR 12, pp. 2825-2830.
3. Abadi, M. et al. (2016). *TensorFlow: A System for Large-Scale Machine Learning*. OSDI 2016.
4. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). *"Why Should I Trust You?": Explaining the Predictions of Any Classifier*. KDD 2016.
5. Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5-32.

---

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{tan2024irishml,
  author = {Tan, Pei Wen (Wendy)},
  title = {Machine Learning Models for Irish Public Transport Delay Prediction},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/wendytnn/irish-transport-ml-models}}
}
```

---

## Future Enhancements

Potential improvements for this machine learning component:

1. **Hyperparameter Tuning**: Grid search/Bayesian optimization for Random Forest
2. **Ensemble Methods**: Stacking/blending multiple models
3. **Feature Selection**: Recursive Feature Elimination (RFE)
4. **Class Imbalance**: SMOTE or class weighting techniques
5. **Time Series Models**: ARIMA/Prophet for temporal patterns
6. **Model Deployment**: REST API with Flask/FastAPI
7. **MLOps Pipeline**: Model versioning, monitoring, retraining
8. **Real-time Inference**: Stream processing with Kafka/Spark

---

## Support

For questions or issues:

1. **Open an Issue**: [GitHub Issues](https://github.com/wendytnn/irish-transport-ml-models/issues)
2. **LinkedIn**: [Wendy Tan Pei Wen](https://www.linkedin.com/in/wendytanpeiwen513/)

