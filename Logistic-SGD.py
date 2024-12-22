import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score


breast_cancer = datasets.load_breast_cancer()

results = {"SGDClassifier with shuffle": [], "SGDClassifier without shuffle": []}

class CustomSGDClassifier(SGDClassifier):
    def __init__(self, loss="hinge", penalty="l2", eta0=0.1, learning_rate="constant", shuffle=True, random_state=None,
                 max_iter=10000):
        super().__init__(loss=loss, penalty=penalty, eta0=eta0, learning_rate=learning_rate, shuffle=shuffle,
                         random_state=random_state, max_iter=max_iter)

    def fit(self, X, y, coef_init=None, intercept_init=None):
        if self.shuffle:
            indices = np.random.permutation(len(X))
            X = X[indices]
            y = y[indices]
        return super().fit(X, y, coef_init, intercept_init)

X_train, X_test, y_train, y_test = train_test_split(breast_cancer.data, breast_cancer.target, test_size=0.25,
                                                    random_state=42, shuffle=False)


print("\nSGDClassifier with shuffle:")

learning_rate = "constant"  
for loss in ["hinge", "log_loss", "modified_huber", "squared_hinge", "perceptron"]:
    for shuffle in [True]:
        model = CustomSGDClassifier(loss=loss, penalty="l2", eta0=0.1, learning_rate=learning_rate, shuffle=shuffle,
                                    random_state=42, max_iter=10000)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        results["SGDClassifier with shuffle"].append(accuracy)

X_train, X_test, y_train, y_test = train_test_split(breast_cancer.data, breast_cancer.target, test_size=0.25,
                                                    random_state=42)

print("\nSGDClassifier without shuffle:")

learning_rate = "constant"  
for loss in ["hinge", "log_loss", "modified_huber", "squared_hinge", "perceptron"]:
    for shuffle in [False]:
        model = CustomSGDClassifier(loss=loss, penalty="l2", eta0=0.1, learning_rate=learning_rate, shuffle=shuffle,
                                    random_state=42, max_iter=10000)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        results["SGDClassifier without shuffle"].append(accuracy)

plt.figure(figsize=(10, 6))
plt.bar(results.keys(), results.values())
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Accuracy of Models with Shuffling and No Shuffling")
plt.show()
