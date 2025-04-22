import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load model yang sudah disimpan
with open("best_model.pkl", "rb") as file:
    model = pickle.load(file)

# Label encoder harus sesuai dengan saat training
gender_encoder = LabelEncoder()
education_encoder = LabelEncoder()
home_encoder = LabelEncoder()
intent_encoder = LabelEncoder()

# Fit encoder sesuai data training
gender_encoder.fit(['Male', 'Female'])
education_encoder.fit(['High School', 'College', 'Graduate', 'Other'])
home_encoder.fit(['Rent', 'Own', 'Mortgage', 'Other'])
intent_encoder.fit(['Personal', 'Medical', 'Education', 'Venture', 'Home Improvement', 'Debt Consolidation'])

# Data baru yang akan diprediksi
new_data = pd.DataFrame([{
    'person_age': 30,
    'person_gender': gender_encoder.transform(['Male'])[0],
    'person_education': education_encoder.transform(['Graduate'])[0],
    'person_income': 45000,
    'person_emp_exp': 5,
    'person_home_ownership': home_encoder.transform(['Rent'])[0],
    'loan_amnt': 10000,
    'loan_intent': intent_encoder.transform(['Personal'])[0],
    'loan_int_rate': 11.5,
    'loan_percent_income': 0.22,
    'cb_person_cred_hist_length': 6,
    'credit_score': 730,
    'previous_loan_defaults_on_file': 0
}])

# Lakukan prediksi
prediction = model.predict(new_data)[0]

# Mapping hasil
status_map = {1: 'Disetujui', 0: 'Ditolak'}
print("Hasil Prediksi Pinjaman:", status_map[prediction])
