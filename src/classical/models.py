from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
import time

class ClassicalModel:
    def __init__(self, model_type):
        self.model_type = model_type
        if model_type == "C_SVM":
            self.model = SVC(probability=True)
        elif model_type == "C_RF":
            self.model = RandomForestClassifier()
        elif model_type == "C_LinReg":
            self.model = LinearRegression()
        elif model_type == "C_SVR":
            self.model = SVR()
        else:
            raise ValueError(f"Unknown classical model: {model_type}")

    def train(self, X_train, y_train):
        start_time = time.time()
        self.model.fit(X_train, y_train)
        end_time = time.time()
        return end_time - start_time

    def evaluate(self, X_test, y_test):
        start_time = time.time()
        score = self.model.score(X_test, y_test)
        end_time = time.time()
        return score, end_time - start_time
