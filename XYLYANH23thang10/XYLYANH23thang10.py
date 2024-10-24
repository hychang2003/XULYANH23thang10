# Import th? vi?n
import time
import numpy as np
from sklearn import svm, neighbors, tree
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from keras.datasets import cifar10
from sklearn.preprocessing import StandardScaler

# 1. Chu?n b? d? li?u (CIFAR-10)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Chuy?n ??i thành vector 1D và chu?n hóa
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Ch?n m?u nh? ?? t?c ?? nhanh h?n
x_train, x_test, y_train, y_test = x_train[:5000], x_test[:1000], y_train[:5000], y_test[:1000]

# 2. Hàm ?o th?i gian và hu?n luy?n mô hình
def evaluate_model(model, x_train, y_train, x_test, y_test):
    start_time = time.time()
    model.fit(x_train, y_train.ravel())  # Hu?n luy?n mô hình
    y_pred = model.predict(x_test)       # D? ?oán
    end_time = time.time()
    
    # 3. Tính các ?? ?o
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    
    return accuracy, precision, recall, end_time - start_time

# 4. SVM
svm_model = svm.SVC()
accuracy, precision, recall, exec_time = evaluate_model(svm_model, x_train, y_train, x_test, y_test)
print(f"SVM - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, Time: {exec_time} seconds")

# 5. KNN
knn_model = neighbors.KNeighborsClassifier(n_neighbors=5)
accuracy, precision, recall, exec_time = evaluate_model(knn_model, x_train, y_train, x_test, y_test)
print(f"KNN - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, Time: {exec_time} seconds")

# 6. Decision Tree
tree_model = tree.DecisionTreeClassifier()
accuracy, precision, recall, exec_time = evaluate_model(tree_model, x_train, y_train, x_test, y_test)
print(f"Decision Tree - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, Time: {exec_time} seconds")


