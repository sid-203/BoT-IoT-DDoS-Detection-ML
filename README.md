# DDoS Attack Detection in IoT Networks Using Machine Learning

A machine learning-based Intrusion Detection System (IDS) for detecting Distributed Denial-of-Service (DDoS) attacks in resource-constrained IoT networks. This research project evaluates lightweight supervised learning models on the BoT-IoT dataset with a focus on achieving high detection accuracy while maintaining minimal false positive rates suitable for IoT deployment.

## Table of Contents

- [Research Question](#research-question)
- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Models Evaluated](#models-evaluated)
- [Results](#results)
- [Key Findings](#key-findings)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Limitations and Future Work](#limitations-and-future-work)
- [Academic Context](#academic-context)
- [References](#references)
- [Author](#author)

## Research Question

**"How accurately can supervised machine learning models detect DDoS attacks in the BoT-IoT dataset while keeping false positive rates low enough for use in resource-constrained IoT networks?"**

This research addresses the critical challenge of deploying effective intrusion detection in IoT environments where devices have limited processing power, memory, and energy budgets.

## Project Overview

The rapid expansion of the Internet of Things (IoT) has led to billions of interconnected devices in homes, industry, healthcare, and critical infrastructure. While this connectivity enables automation and new digital services, it also significantly enlarges the attack surface of modern networks. Among various threats, **Distributed Denial-of-Service (DDoS) attacks** are particularly disruptive in IoT environments.

### Key Objectives

1. **Evaluate lightweight ML models** suitable for resource-constrained IoT gateways
2. **Minimize false positive rates** to prevent blocking legitimate traffic
3. **Compare model complexity vs. performance** trade-offs
4. **Establish best practices** for IoT-based DDoS detection

## Problem Statement

### Challenges in IoT Security

- **Weak Default Configurations**: Many IoT devices deployed with minimal security
- **Limited Security Mechanisms**: Built-in protections often inadequate
- **Poor Patch Management**: Devices rarely updated, creating persistent vulnerabilities
- **Resource Constraints**: Limited CPU, memory, and energy for running complex security solutions

### Why DDoS is Critical for IoT

- IoT devices can be compromised into **botnets** generating massive malicious traffic
- DDoS attacks cause **loss of availability** in critical services
- Traditional signature-based IDS struggle with **evolving attack patterns**
- False positives in resource-constrained environments lead to:
  - Legitimate traffic being blocked
  - Unnecessary resource consumption
  - Reduced administrator trust in the system

## Dataset

### BoT-IoT Dataset

**Source**: Simulated IoT network with legitimate devices and compromised hosts

**Subset Used**: `data_t.csv`
- **Total Records**: 1,109 network flows
- **DDoS Traffic**: 1,044 flows (94.1%)
- **Normal Traffic**: 65 flows (5.9%)

**Train/Test Split**: 70/30 stratified split
- **Training Set**: 776 flows (731 DDoS, 45 Normal)
- **Test Set**: 333 flows (313 DDoS, 20 Normal)

### Features

The dataset contains **35 flow-level features** across multiple categories:

**Flow Statistics:**
- `pkts`, `bytes`, `spkts`, `dpkts`, `sbytes`, `dbytes`

**Timing Information:**
- `stime`, `ltime`, `dur`, `mean`, `stddev`, `min`, `max`

**Rate Features:**
- `rate`, `srate`, `drate`

**Protocol Information:**
- `proto` (tcp, udp, arp, etc.)
- `flgs` (TCP flags)
- `state` (connection state)

**Addressing:**
- `saddr` (source address)
- `daddr` (destination address)
- `sport`, `dport` (ports)

**Labels:**
- `attack` (binary flag)
- `category` (DDoS or Normal)
- `subcategory` (attack subtype)

## Methodology

### Preprocessing Pipeline

1. **Label Encoding**
   - Converted `category` field to binary labels
   - DDoS → 0 (positive class)
   - Normal → 1 (negative class)

2. **Feature Engineering**
   - Removed label fields (`attack`, `category`, `subcategory`) to prevent data leakage
   - One-hot encoded categorical features: `proto`, `flgs`, `state`, `saddr`, `daddr`
   - Dropped all-NaN columns

3. **Missing Value Handling**
   - Applied median imputation using `SimpleImputer`
   - Robust to outliers while preserving distribution

4. **Feature Scaling**
   - Standardized all features to zero mean and unit variance
   - Essential for Logistic Regression and gradient-based models

5. **Data Splitting**
   - 70/30 stratified train-test split
   - Preserved class distribution in both sets

### Evaluation Metrics

- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **False Positive Rate (FPR)**: False positives / (False positives + True negatives)
- **Confusion Matrix**: Detailed classification breakdown
- **ROC-AUC**: Area Under the Receiver Operating Characteristic curve

## Models Evaluated

### 1. Logistic Regression

**Type**: Linear classifier

**Advantages:**
- Extremely lightweight (simple dot product + sigmoid)
- Low training and inference cost
- Ideal for resource-constrained IoT gateways
- Provides probability estimates

**Configuration:**
- `max_iter=200` for convergence
- Default L2 regularization

**Use Case**: Best candidate for edge deployment on IoT devices

### 2. Decision Tree

**Type**: Non-linear rule-based classifier

**Advantages:**
- Interpretable "if-then" decision paths
- No feature scaling required
- Can capture non-linear patterns
- Relatively cheap inference

**Configuration:**
- Gini impurity criterion
- Default hyperparameters

**Use Case**: Good balance between interpretability and performance

### 3. Random Forest

**Type**: Ensemble of decision trees

**Advantages:**
- Strong generalization through bootstrapping
- Feature importance analysis
- Robust to overfitting
- Handles non-linear relationships well

**Configuration:**
- `n_estimators=100` (default)
- Bootstrap sampling with feature subsampling

**Use Case**: Baseline for maximum performance comparison

## Results

### Model Performance Summary

| Model | Accuracy | Precision | Recall | F1-Score | FPR | Training Cost |
|-------|----------|-----------|--------|----------|-----|---------------|
| **Logistic Regression** | 100% | 100% | 100% | 100% | 0% | Lowest |
| **Decision Tree** | 100% | 100% | 100% | 100% | 0% | Low |
| **Random Forest** | 100% | 100% | 100% | 100% | 0% | Moderate |

### Detailed Results (Test Set: 313 DDoS, 20 Normal)

```
Logistic Regression:
- True Positives (DDoS detected): 313
- True Negatives (Normal detected): 20
- False Positives (Normal flagged as DDoS): 0
- False Negatives (DDoS missed): 0
- FPR on Normal Traffic: 0.000

Decision Tree:
- True Positives (DDoS detected): 313
- True Negatives (Normal detected): 20
- False Positives (Normal flagged as DDoS): 0
- False Negatives (DDoS missed): 0
- FPR on Normal Traffic: 0.000

Random Forest:
- True Positives (DDoS detected): 313
- True Negatives (Normal detected): 20
- False Positives (Normal flagged as DDoS): 0
- False Negatives (DDoS missed): 0
- FPR on Normal Traffic: 0.000
```

### ROC-AUC Analysis

All three models achieved **perfect ROC curves** with AUC = 1.000, indicating complete separability between DDoS and Normal classes in this dataset subset.

### Feature Importance (Random Forest)

**Top 10 Most Important Features:**

1. `pkSeqID` - Packet sequence identifier
2. `ltime` - Last packet time
3. `daddr_192.168.100.3` - Specific destination IP
4. `stime` - Start time
5. `seq` - Sequence number
6. `pkts` - Total packets
7. `dur` - Flow duration
8. `proto_udp` - UDP protocol indicator
9. `rate` - Packet rate
10. `proto_tcp` - TCP protocol indicator

**Key Insight**: Flow timing, packet counts, and rate-based features dominate the decision process, but specific IP addresses also contribute significantly (indicating potential overfitting to testbed).

## Key Findings

### 1. Lightweight Models Suffice for BoT-IoT

**Critical Discovery**: Logistic Regression (the simplest model) matched Random Forest performance perfectly.

**Implications:**
- Classes are **linearly separable** in the feature space
- No benefit from complex ensemble methods
- **Logistic Regression is ideal** for IoT gateway deployment
- Computational overhead of Random Forest is unjustified

### 2. Perfect Metrics Indicate Data Characteristics

**Why 100% accuracy?**
- BoT-IoT DDoS traffic has **highly distinctive patterns**
- Extreme differences in packet rates, timing, and volume
- Limited variety in attack and normal traffic patterns

### 3. False Positive Rate = 0

**Highly desirable for IoT deployments:**
- No legitimate traffic would be blocked
- No unnecessary resource consumption
- Maintains user/administrator trust

### 4. Comparison with Prior Research

**Consistent with Literature:**
- Pokhrel et al. (2021): KNN achieved 92-99% accuracy with SMOTE balancing
- Alosaimi & Almutairi (2023): Decision Tree/Bagging achieved ~100% accuracy
- Ashraf et al. (2025): Random Forest achieved 99.2% accuracy

**Novel Contributions:**
- **Explicit FPR focus** as primary constraint
- **Direct comparison** of lightweight vs. complex models
- **DDoS-specific** rather than general intrusion detection
- **Resource constraint awareness** in model selection

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager
- Google Colab (optional, for replicating exact environment)

### Required Libraries

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Setup Instructions

1. **Clone the repository** (or download files)
   ```bash
   git clone https://github.com/yourusername/ddos-iot-detection.git
   cd ddos-iot-detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the dataset**
   - Place `data_t.csv` in the project directory
   - Or download BoT-IoT dataset and create subset

4. **Run the analysis**
   ```bash
   # Open Jupyter notebook
   jupyter notebook Net_Sec.ipynb
   
   # Or run Python script
   python ddos_detection.py
   ```

## Usage

### Training Models

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load data
data = pd.read_csv('data_t.csv')

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data["category"])

# Prepare features (see full preprocessing in notebook)
# ... preprocessing steps ...

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2%}")
```

### Making Predictions

```python
# Predict on new flow data
new_flow = preprocess_flow(flow_data)  # Apply same preprocessing
prediction = model.predict([new_flow])

if prediction == 0:
    print("⚠️ DDoS Attack Detected!")
    # Trigger mitigation (rate limiting, blocking, etc.)
else:
    print("✓ Normal Traffic")
```

## Project Structure

```
ddos-iot-detection/
│
├── Net_Sec.ipynb                      # Main Jupyter notebook
├── data_t.csv                         # BoT-IoT dataset subset
├── ST20238021_CIS7037_PORT1.pdf       # Full research report
├── README.md                          # This file
│
├── models/                            # Saved trained models
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   └── random_forest.pkl
│
├── results/                           # Generated outputs
│   ├── confusion_matrices/
│   ├── roc_curves/
│   ├── feature_importance.png
│   └── performance_metrics.csv
│
└── requirements.txt                   # Python dependencies
```

## Limitations and Future Work

### Current Limitations

#### 1. Dataset Imbalance

**Issue**: The dataset is **heavily imbalanced** (94.1% DDoS, 5.9% Normal)

**Impact on Results**:
- Perfect metrics may be **misleading**
- False Positive Rate estimated from only **20 normal samples** in test set
- A single misclassification would change FPR by 5%
- **Not representative** of real-world IoT traffic distributions

**Acknowledgment**: The current results (100% accuracy, 0% FPR) are **artificially inflated** due to class imbalance and limited normal traffic diversity.

#### 2. Overfitting to Testbed

**Issue**: One-hot encoding of IP addresses allows models to memorize specific hosts

**Evidence**:
- `daddr_192.168.100.3` appears as a top feature
- Models learn testbed-specific patterns
- May not generalize to other networks

#### 3. Limited Normal Traffic Diversity

**Issue**: Only 65 normal flows total
- Insufficient to represent real IoT traffic variety
- Homogeneous traffic patterns
- No variability in legitimate device behavior

#### 4. Synthetic Dataset

**Issue**: BoT-IoT is a simulated environment
- Fixed IP ranges
- Scripted attack patterns
- May not reflect real-world complexity

### Future Improvements

#### 1. Dataset Balancing (Priority)

**Planned Solutions**:

**Option A - SMOTE Oversampling:**
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)
```

**Option B - Undersampling DDoS:**
- Randomly downsample majority class
- Create balanced 50/50 split
- Preserve all minority class samples

**Option C - Hybrid Approach:**
- Combine SMOTE + random undersampling
- Achieve better distribution
- Reduce overfitting risk

**Expected Impact**:
- More realistic performance metrics
- Better FPR estimates
- Improved generalization

#### 2. Cross-Dataset Validation

**Additional Datasets to Test:**
- **ToN-IoT**: Heterogeneous IoT/IIoT network dataset
- **CICIoT2023**: Recent IoT attack dataset
- **N-BaIoT**: Real IoT device traffic
- **UNSW-NB15**: Includes modern attack vectors

**Benefits**:
- Validate model generalization
- Test on diverse traffic patterns
- Identify dataset-specific biases

#### 3. Feature Engineering Improvements

**Remove IP Address Encoding:**
- Aggregate to subnet level instead
- Use IP-agnostic features only
- Improve cross-network generalization

**Add Network-Level Features:**
- Protocol distribution statistics
- Port usage patterns
- Flow clustering analysis

#### 4. Advanced Modeling Techniques

**Incremental Learning:**
- Online learning for concept drift
- Adaptive to evolving attack patterns
- Suitable for long-running IoT deployments

**Ensemble Methods:**
- Combine multiple weak learners
- Weighted voting schemes
- Adaptive ensemble selection

**Deep Learning Exploration:**
- LSTM for sequence analysis
- CNN for traffic pattern recognition
- Autoencoders for anomaly detection

#### 5. Real-World Deployment Considerations

**Hardware Profiling:**
- Benchmark on actual IoT gateways
- Measure memory footprint
- Calculate inference latency
- Energy consumption analysis

**Integration with Network Stack:**
- Develop packet capture module
- Real-time flow aggregation
- Mitigation trigger system

**Hybrid Defense Architecture:**
```
[IoT Devices] → [Gateway: Logistic Regression IDS]
                        ↓
                 [Alert on Suspicious]
                        ↓
              [Cloud: Deep Analysis with Random Forest]
                        ↓
                [Coordinated Mitigation]
```

## Best Practices for IoT DDoS Detection

Based on this research, the following best practices are recommended:

### 1. Start Simple, Scale if Needed

- Begin with **Logistic Regression** for edge deployment
- Monitor performance in production
- Only escalate to complex models if necessary

### 2. Prioritize False Positive Rate

- Explicitly design and evaluate around **low FPR**
- False positives are costly in resource-constrained environments
- Balance detection rate with operational impact

### 3. Use Flow-Level Features

- Standard flow statistics (packets, bytes, rates) are sufficient
- No need for sophisticated handcrafted features
- Enables real-time processing

### 4. Avoid Label Leakage

- Strictly separate label fields from features
- Document preprocessing decisions clearly
- Validate feature independence

### 5. Handle Class Imbalance Deliberately

- Don't rely on accuracy alone
- Use resampling techniques (SMOTE, undersampling)
- Report FPR and confusion matrices explicitly

### 6. Combine ML with Network Controls

- ML for detection and flagging
- Trigger network-level mitigation:
  - Rate limiting
  - Connection throttling
  - Upstream filtering
  - Blacklisting

### 7. Monitor and Recalibrate

- Periodic retraining on recent traffic
- Monitor FPR in production
- Prevent model drift and alert fatigue

## Academic Context

**Module**: CIS7037 - Network Security  
**Institution**: Cardiff Metropolitan University  
**Program**: MSc Advanced Cyber Security  
**Academic Year**: 2023-2024

### Research Contributions

1. **DDoS-Specific Focus**: Unlike prior work treating DDoS as one of many attacks
2. **FPR as Primary Constraint**: Explicit false positive optimization for IoT
3. **Lightweight Model Emphasis**: Direct comparison for resource-constrained deployment
4. **Honest Limitation Reporting**: Transparent about dataset constraints and overfitting

### Video Presentation

**Link**: [https://youtu.be/bRVmAoqZG7k](https://youtu.be/bRVmAoqZG7k)

## References

1. Sinha, S. (2024). State of IoT 2024: Number of connected IoT devices growing 13% to 18.8 billion globally. IoT Analytics.

2. Alosaimi, S. and Almutairi, S.M. (2023). An Intrusion Detection System Using BoT-IoT. Applied Sciences, 13(9), p.5427. https://doi.org/10.3390/app13095427

3. Ashraf, J., Raza, G.M., Kim, B.-S., Wahid, A. and Kim, H.-Y. (2025). Making a Real-Time IoT Network Intrusion-Detection System (INIDS) Using a Realistic BoT–IoT Dataset with Multiple Machine-Learning Classifiers. Applied Sciences, 15(4), p.2043. https://doi.org/10.3390/app15042043

4. Pokhrel, S., Abbas, R. and Aryal, B. (2021). IoT Security: Botnet detection in IoT using Machine learning. arXiv:2104.02231 [cs]. https://arxiv.org/abs/2104.02231

5. Kumari, P. and Jain, A.K. (2023). A Comprehensive Study of DDoS Attacks over IoT Network and Their Countermeasures. Computers & Security, 127, p.103096. https://doi.org/10.1016/j.cose.2023.103096

## Author

**Sid Ali Bendris**  
MSc Advanced Cyber Security  
Cardiff Metropolitan University

**Module Leader**: Dr Sheikh Tahir Bakhsh

## License

This project is developed for academic purposes as part of the MSc Advanced Cyber Security program at Cardiff Metropolitan University.

## Acknowledgments

- Cardiff Metropolitan University for academic guidance
- BoT-IoT dataset creators for providing realistic IoT attack data
- Research community for prior work on IoT intrusion detection
- Scikit-learn and Python data science ecosystem

---

**Project Status**: Active Research - Improvements Planned  
**Last Updated**: February 2026  
**Version**: 1.0.0 (Initial Research Implementation)

**Next Steps**: Dataset rebalancing and cross-validation on additional IoT datasets
