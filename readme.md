# Breast Cancer & Iris Dataset Classification Using Logistic Regression & SGDClassifier

## Introduction
This project utilizes **Logistic Regression** and **SGDClassifier** to classify breast cancer (benign vs. malignant) and Iris flower species (setosa, versicolor, virginica). By analyzing the impact of various parameters and aims to identify optimal configurations for accuracy and efficiency.

## Datasets
1. **Breast Cancer Dataset**:
   - Features: Texture, width, size, smoothness, etc.
   - Target: Benign or malignant tumors.

2. **Iris Dataset**:
   - Features: Sepal/petal length and width.
   - Target: Iris species (setosa, versicolor, virginica).

## Models & Techniques
1. **Logistic Regression**:
   - Regularization schemes: L1, L2, Elastic Net.
   - Solvers: Liblinear, Saga.

2. **SGDClassifier**:
   - Parameters:
     - Learning rates: Constant, optimal, adaptive.
     - Loss functions: Hinge, log loss, perceptron, etc.
     - Penalties: L1, L2, Elastic Net.
     - Solvers: Liblinear, Saga.
     - Shuffling: True/False.

## Training and Evaluation
- Data Split: 75% training and 25% testing.
- Metrics: Accuracy.
- Input combinations for testing include learning rates, penalties, solvers, and shuffle configurations.

## Results & Analysis
Key experiments were conducted with combinations of parameters, and their results are summarized below:

### Experiment 1: Iris Dataset - Optimal Learning Rate
**Command**:
```python
run(datasets.load_iris(), "optimal", "hinge", True, "elasticnet", "saga")
```
**Analysis**:
- Elastic Net penalty improved feature selection and prevented overfitting.
- The Saga solver enabled efficient optimization, enhancing accuracy.

---

### Experiment 2: Breast Cancer Dataset - Adaptive Learning Rate
**Command**:
```python
run(datasets.load_breast_cancer(), "adaptive", "hinge", True, "elasticnet", "saga")
```
**Analysis**:
- Adaptive learning rate dynamically adjusted optimization in this case.
- Elastic Net regularization and Saga solver achieved a robust performance.

---

### Experiment 3: Iris Dataset - Constant Learning Rate
**Command**:
```python
run(datasets.load_iris(), "constant", "hinge", False, "l2", "liblinear")
```
**Analysis**:
- Ridge (L2) regularization controlled complexity and prevented overfitting.
- The lack of shuffling led to potential model bias.

---

### Experiment 4: Breast Cancer Dataset - Perceptron Loss
**Command**:
```python
run(datasets.load_breast_cancer(), "optimal", "perceptron", True, "elasticnet", "saga")
```
**Analysis**:
- Perceptron loss performed well with the linearly separable data.
- Elastic Net provided flexibility, while Saga solver ensured convergence.

---

## Conclusion
- **SGDClassifier**: The performance is highly sensitive to parameter tuning, such as loss functions, penalties, and learning rates.
- **Logistic Regression**: Demonstrated consistent results with slight performance changes across configurations.