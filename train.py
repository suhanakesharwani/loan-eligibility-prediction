import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler

df = pd.read_csv('data/loan_data.csv') 

# Preprocessing
le = LabelEncoder()
for col in ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']:
    df[col] = le.fit_transform(df[col].astype(str))

# Convert '3+' to 3 and handle missing values if any
df['Dependents'] = df['Dependents'].replace('3+', 3).fillna(0).astype(int)

# --- FIX: Match the column names from your Print output ---
df['Total_Income'] = df['Applicant_Income'] + df['Coapplicant_Income']

# Define Features using exact column names
X = df[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
        'Applicant_Income', 'Loan_Amount', 'Loan_Amount_Term', 
        'Credit_History', 'Property_Area', 'Total_Income']]
# Note: I removed Coapplicant_Income from X because Total_Income usually captures it, 
# but keep it if you want the model to see both. 

y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ros = RandomOverSampler(random_state=0)
X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_res_scaled = scaler.fit_transform(X_train_res)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_res_scaled, y_train_res)

# Save
with open('models/scaler_2.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('models/best_loan_model_1.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Models saved with updated column names!")