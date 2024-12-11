# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


class LoanDefaultModel:
    def __init__(self, model):
        self.model = model
        self.scaler = StandardScaler()

    def load_data(self, path):
        # Load data
        data = pd.read_excel(path)
        return data

    def preprocess(self, data):
    # Drop unnecessary columns
        data = data.drop(['customer_id', 'transaction_date'], axis=1)
        
        # Clean and convert the 'term' column
        data['term'] = data['term'].str.extract('(\d+)').astype(int)

        # Encode categorical features
        categorical_cols = ['sub_grade', 'home_ownership', 'purpose', 'application_type', 'verification_status']
        for col in categorical_cols:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
        
        # Scale numerical features
        num_features = ['cibil_score', 'annual_inc', 'loan_amnt', 'int_rate', 'account_bal', 'term']
        data[num_features] = self.scaler.fit_transform(data[num_features])
        
        # Split into features and target
        X = data.drop('loan_status', axis=1)
        y = data['loan_status']
        return X, y


    def train(self, X_train, y_train):
        # Train the model
        self.model.fit(X_train, y_train)

    def test(self, X_test, y_test):
        # Predict and evaluate
        predictions = self.model.predict(X_test)
        print(classification_report(y_test, predictions))
        return accuracy_score(y_test, predictions)

    def predict(self, X):
        # Predict new data
        return self.model.predict(X)


if __name__ == "__main__":
    # Instantiate the class
    logistic_model = LoanDefaultModel(LogisticRegression())

    # Load data
    train_data = logistic_model.load_data('../data/train_data.xlsx')

    # Preprocess data
    X, y = logistic_model.preprocess(train_data)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    

    # Train and test model
    logistic_model.train(X_train, y_train)
    accuracy = logistic_model.test(X_test, y_test)
    print("Logistic Regression Accuracy:", accuracy)
    
    rf_model = LoanDefaultModel(RandomForestClassifier(n_estimators=100, random_state=42))

    # Train and test Random Forest
    rf_model.train(X_train, y_train)
    rf_accuracy = rf_model.test(X_test, y_test)
    print("Random Forest Accuracy:", rf_accuracy)
