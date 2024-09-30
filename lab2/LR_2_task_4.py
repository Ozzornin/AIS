import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Load the dataset from the input file
data_path = "income_data.txt"
column_names = [
    "Age",
    "Workclass",
    "fnlwgt",
    "Education",
    "Education_Num",
    "Marital_Status",
    "Occupation",
    "Relationship",
    "Race",
    "Sex",
    "Capital_Gain",
    "Capital_Loss",
    "Hours_Per_Week",
    "Native_Country",
    "Income",
]
data = pd.read_csv(data_path, sep=",", header=None, names=column_names)

# Label encode categorical columns
categorical_columns = [
    "Workclass",
    "Education",
    "Marital_Status",
    "Occupation",
    "Relationship",
    "Race",
    "Sex",
    "Native_Country",
    "Income",
]

# Encode categorical features
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
for column in categorical_columns:
    data[column] = encoder.fit_transform(data[column])

# Separate features and target
features = data.drop("Income", axis=1)
target = data["Income"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.3, random_state=42
)

# Define models for evaluation, using OneVsRestClassifier for Logistic Regression
classifiers = {
    "Logistic Regression": OneVsRestClassifier(LogisticRegression(solver="liblinear")),
    "Linear Discriminant": LinearDiscriminantAnalysis(),
    "K-Neighbors Classifier": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Gaussian Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(gamma="scale"),
}

# Perform cross-validation and evaluate models
for classifier_name, model in classifiers.items():
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring="accuracy")
    print(
        f"{classifier_name}: Mean Accuracy = {scores.mean():.4f}, Std Dev = {scores.std():.4f}"
    )
