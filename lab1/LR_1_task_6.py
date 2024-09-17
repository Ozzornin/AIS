import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from utilities import visualize_classifier

# Завантажимо дані
data = pd.read_csv(
    "data_multivar_nb.txt", sep=","
) 
X = data.iloc[:, :-1].values  # Ознаки
y = data.iloc[:, -1].values  # Цільові значення

# Розділимо на тренувальні та тестові дані
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

# Класифікація за допомогою SVM
svm_classifier = SVC(kernel="linear", random_state=42)
svm_classifier.fit(X_train, y_train)

# Прогнозуємо на тестових даних
y_pred_svm = svm_classifier.predict(X_test)
visualize_classifier(svm_classifier, X_test, y_test, "SVM")
# Розраховуємо метрики якості
print("SVM Classifier Metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Classification Report:\n", classification_report(y_test, y_pred_svm))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))

# Класифікація за допомогою наївного Байєсівського класифікатора
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# Прогнозуємо на тестових даних
y_pred_nb = nb_classifier.predict(X_test)
visualize_classifier(nb_classifier, X_test, y_test, "NB")
# Розраховуємо метрики якості
print("\nNaive Bayes Classifier Metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print("Classification Report:\n", classification_report(y_test, y_pred_nb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_nb))
