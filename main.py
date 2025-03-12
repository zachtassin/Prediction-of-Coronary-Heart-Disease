import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


class CustomLogisticRegression:
    def __init__(self, learning_rate=0.1):
        self.weights = None
        self.bias = 0
        self.learning_rate = learning_rate
        self.train_accuracies = []
        self.losses = []

    def _sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def compute_loss(self, y_true, y_pred):
        y_zero_loss = y_true * np.log(y_pred + 1e-9)
        y_one_loss = (1 - y_true) * np.log(1 - y_pred + 1e-9)
        return -np.mean(y_zero_loss + y_one_loss)

    def compute_gradients(self, x, y_true, y_pred):
        difference = y_pred - y_true
        gradient_b = np.mean(difference)
        gradients_w = np.dot(x.T, difference) / len(y_true)
        return gradients_w, gradient_b

    def update_model_parameters(self, error_w, error_b):
        self.weights -= self.learning_rate * error_w
        self.bias -= self.learning_rate * error_b

    def fit(self, x, y, epochs=150):
        x = np.array(x)
        y = np.array(y)

        self.weights = np.zeros(x.shape[1])
        self.bias = 0

        for _ in range(epochs):
            linear_model = np.dot(x, self.weights) + self.bias
            predictions = self._sigmoid(linear_model)
            loss = self.compute_loss(y, predictions)

            error_w, error_b = self.compute_gradients(x, y, predictions)
            self.update_model_parameters(error_w, error_b)

            predicted_classes = [1 if p > 0.5 else 0 for p in predictions]
            self.train_accuracies.append(accuracy_score(y, predicted_classes))
            self.losses.append(loss)

    def predict(self, x):
        linear_model = np.dot(x, self.weights) + self.bias
        probabilities = self._sigmoid(linear_model)
        return np.array([1 if p > 0.5 else 0 for p in probabilities])



def load_data():
    dataset = load_breast_cancer()
    x = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = pd.Series(dataset.target, name="target")
    return train_test_split(x, y, test_size=0.2, random_state=42)



if __name__ == "__main__":
    x_train, x_test, y_train, y_test = load_data()


    x_train = x_train.astype(float)
    x_test = x_test.astype(float)


    custom_lr = CustomLogisticRegression()
    custom_lr.fit(x_train, y_train, epochs=150)


    custom_predictions = custom_lr.predict(x_test)
    custom_accuracy = accuracy_score(y_test, custom_predictions)
    print(f"Custom Logistic Regression Accuracy: {custom_accuracy:.4f}")


    sklearn_lr = LogisticRegression(solver='newton-cg', max_iter=150)
    sklearn_lr.fit(x_train, y_train)
    sklearn_predictions = sklearn_lr.predict(x_test)
    sklearn_accuracy = accuracy_score(y_test, sklearn_predictions)
    print(f"Scikit-learn Logistic Regression Accuracy: {sklearn_accuracy:.4f}")
