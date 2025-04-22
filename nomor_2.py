import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import pickle

class BestLoanModel:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_and_preprocess(self):
        self.df = pd.read_csv(self.csv_path)


        # Encode fitur kategorikal
        categorical_cols = ["person_gender", "person_education", "person_home_ownership", "loan_intent", "previous_loan_defaults_on_file"]
        le = LabelEncoder()
        for col in categorical_cols:
            self.df[col] = le.fit_transform(self.df[col])

        # Split fitur dan target
        X = self.df.drop('loan_status', axis=1)
        y = self.df['loan_status']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    def train_model(self):
        # Gunakan model terbaik yang telah dipilih
        self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        predictions = self.model.predict(self.X_test)
        acc = accuracy_score(self.y_test, predictions)
        report = classification_report(self.y_test, predictions)
        print(f"Akurasi Model Terbaik: {acc}\n")
        print("Classification Report:\n", report)

    def save_model(self, filename='best_model_oop.pkl'):
        with open(filename, 'wb') as file:
            pickle.dump(self.model, file)
        print(f"Model disimpan dalam file: {filename}")


# Eksekusi
if __name__ == "__main__":
    model = BestLoanModel("Dataset_A_loan.csv")
    model.load_and_preprocess()
    model.train_model()
    model.evaluate_model()
    model.save_model()
