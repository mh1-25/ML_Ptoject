import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=["label"]).values
    y_train = train_df["label"].values

    X_test = test_df.drop(columns=["label"]).values
    y_test = test_df["label"].values

    return X_train, y_train, X_test, y_test

def preprocess(X_train, X_test):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    return X_train, X_test, sc

def train_logistic_regression(X_train, y_train):
    classifier = LogisticRegression(max_iter=500, solver='lbfgs')
    classifier.fit(X_train, y_train)

    return classifier

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    return acc, cm

def main():
    
    train_path = "Images/Dataset/train.csv"
    test_path = "Images/Dataset/test.csv"

    print("Loading dataset...")
    X_train, y_train, X_test, y_test = load_data(train_path, test_path)

    print("Preprocessing...")
    X_train, X_test, sc = preprocess(X_train, X_test)

    print ("Training Logistic Regression...")
    model = train_logistic_regression(X_train, y_train)

    print("Evaluating...")
    acc, cm = evaluate(model, X_test, y_test)

    print(f"Accuracy: {acc}")
    print("Confusion Matrix:")
    print(cm)

if __name__ == "__main__":
    main()