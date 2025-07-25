# 📊 ML Project Portfolio

This repository contains three end-to-end machine learning projects demonstrating expertise in classification, regression, and computer vision. Each project includes preprocessing, model training, evaluation, and ethical considerations.

---

## 🧩 Project 1 – Binary Classification: Credit Card Fraud Detection

### 🔍 Objective
Detect fraudulent credit card transactions using supervised learning techniques.

### 📂 Dataset
- **Type**: Financial, anonymized transaction records
- **Size**: ~285,000 rows, binary labels (fraud / not fraud)

### 🔧 Models Used
- Logistic Regression
- Linear Discriminant Analysis (LDA)

### 📊 Evaluation Metrics
- Cross-Validation Accuracy: `0.9983` (Logistic Regression)
- ROC-AUC Score: `0.9999`
- Confusion Matrix Analysis

### ✅ Outcome
Logistic Regression showed better performance across metrics and faster training, making it the preferred choice for production.

---

## 📈 Project 2 – Regression: Financial Forecasting

### 🔍 Objective
Predict a continuous financial variable (e.g., price, volume) based on market or firm-level features.

### 📂 Dataset
- **Type**: Structured financial data
- **Target Variable**: Continuous (e.g., future stock value)
- **Features**: Economic indicators, historical prices, volumes

### 🔧 Models Used
- Linear Regression
- Lasso Regression
- ElasticNet Regression

### 📊 Evaluation Metrics
- RMSE
- R² Score

### ✅ Outcome
ElasticNet offered the best balance between predictive performance and regularization, reducing overfitting on test data.

---

## 🖼️ Project 3 – Multiclass Image Classification: Visual Recognition

### 🔍 Objective
Classify images into one of several categories using deep learning.

### 📂 Dataset
- **Type**: Images (e.g., animals, objects, or digits)
- **Format**: JPEG/PNG
- **Labels**: Multiple classes (e.g., 10–100 classes)

### 🧠 Techniques
- Data Augmentation
- CNN Architecture (e.g., custom or transfer learning)
- Softmax Output for Multiclass Prediction

### 📊 Evaluation Metrics
- Accuracy
- Precision/Recall per class
- Confusion Matrix

### ✅ Outcome
Model achieved high classification accuracy using convolutional layers, batch normalization, and regularization. Optimized with cross-entropy loss and early stopping.

---

## 🔐 Ethical Considerations
- **Bias & Fairness**: Ensuring no demographic bias in classification tasks
- **Privacy**: Handling sensitive financial and image data securely
- **Model Risk**: False negatives in fraud detection can lead to financial loss
- **Monitoring**: Continuous updates needed for evolving fraud/image patterns

---

## 📚 References
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Keras & TensorFlow](https://www.tensorflow.org/)
- Stack Overflow

---

## 🛠️ Tools & Technologies
- Python 3.x
- Pandas, NumPy, Scikit-learn
- Matplotlib, Seaborn
- TensorFlow / Keras (for CNNs)
- Jupyter Notebook

