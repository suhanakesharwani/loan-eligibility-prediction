# 🏦 Loan Eligibility Prediction using Machine Learning

A Machine Learning project developed to predict whether a loan applicant is eligible for approval based on financial, demographic, and property-related attributes.

The system assists banks and financial institutions in automating loan approval decisions using predictive analytics.

---

## 🚀 Project Overview

Loan approval evaluation is traditionally performed through manual verification, which can be time-consuming and inconsistent.  

This project builds an end-to-end Machine Learning pipeline that analyzes applicant information and predicts loan eligibility with high accuracy.

The model learns patterns from historical loan data to identify applicants who are more likely to receive approval.

---

## 🎯 Objective

- Predict loan approval status using applicant data
- Reduce manual screening effort
- Handle real-world challenges such as categorical data and class imbalance
- Compare multiple Machine Learning algorithms

---

## 📊 Dataset Description

The dataset contains information about loan applicants including:

- Personal details
- Employment status
- Education level
- Income information
- Loan amount and duration
- Credit history
- Property area

**Target Variable**
- Loan Status (Approved / Not Approved)

---

## ⚙️ Methodology

### Data Preprocessing
The dataset was cleaned and prepared before model training. Key preprocessing steps included:

- Encoding categorical variables into numerical format
- Handling special values in dependent features
- Separating input features and target variable
- Splitting the dataset into training and testing sets

---

### Handling Class Imbalance
Loan datasets often contain more approved applications than rejected ones.  
Oversampling techniques were applied to balance the training data and prevent model bias.

---

### Feature Scaling
Numerical features were standardized to ensure consistent model performance and faster convergence during training.

---

## 🤖 Machine Learning Models Used

Two classification models were implemented and evaluated:

- **Logistic Regression** — baseline linear classification model
- **Support Vector Machine (RBF Kernel)** — non-linear classifier capable of capturing complex relationships

---

## 📈 Model Evaluation

Models were evaluated using multiple performance metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix analysis

Performance comparison helped identify the best model for loan prediction.

---

## ✅ Results

- Logistic Regression achieved the highest overall accuracy (~80%).
- The model demonstrated strong capability in identifying approved loan applications.
- Support Vector Machine also performed competitively with comparable results.

The comparison shows that simpler models can perform effectively when proper preprocessing and balancing techniques are applied.

---

## 📉 Visualizations

The project includes visual analysis to interpret model performance:

- Confusion Matrix visualization
- Model accuracy comparison charts

These visualizations help understand prediction behavior and classification quality.

---

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Imbalanced-learn
- Matplotlib

---

## 📂 Project Structure

Loan-Eligibility-Prediction/
│
├── data/
├── assets/
├── notebook/
├── requirements.txt
└── README.md

---

## 💡 Key Learnings

- Data preprocessing for real-world datasets
- Handling categorical variables
- Managing imbalanced classification problems
- Comparing Machine Learning models
- Evaluating classification performance

---

## 🔮 Future Improvements

- Hyperparameter tuning
- Feature engineering
- Model deployment using Streamlit
- Explainable AI techniques
- Real-time prediction API

---

## 👩‍💻 Author

**Suhana Kesharwani**  
B.Tech Information Technology  
Machine Learning & AI Enthusiast

---

## ⭐ Support

If you found this project useful:

⭐ Star the repository  
🍴 Fork and explore  
📢 Share with others