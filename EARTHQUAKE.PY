import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def load_data():
    np.random.seed(42)
    n = 300
    df = pd.DataFrame({
        'magnitude': np.random.uniform(2.0, 8.0, n),
        'depth': np.random.uniform(5, 100, n),
        'latitude': np.random.uniform(-10, 10, n),
        'longitude': np.random.uniform(70, 90, n),
        'gap': np.random.uniform(10, 180, n)
    })
    return df


def classify(mag):
    if mag < 3.0:   return 'Minor'
    elif mag < 5.0: return 'Light'
    elif mag < 6.0: return 'Moderate'
    elif mag < 7.0: return 'Strong'
    else:           return 'Major'


def preprocess(df):
    df['Class'] = df['magnitude'].apply(classify)
    X = df[['depth', 'latitude', 'longitude', 'gap']]
    y = df['Class']
    return train_test_split(X, y, test_size=0.2, random_state=42)


def get_models():
    return {
        "Decision Tree": DecisionTreeClassifier(max_depth=5),
        "Random Forest": RandomForestClassifier(),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC()
    }


def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[name] = acc
        print(f"\n{name} Accuracy:", acc)
        print(classification_report(y_test, preds))
    return results


def plot_comparison(results):
    plt.figure()
    plt.bar(results.keys(), results.values())
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.show()


def plot_confusion(model, X_test, y_test):
    preds = model.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=model.classes_)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.show()


def plot_decision_tree(model, X):
    plt.figure(figsize=(14, 8))
    plot_tree(model, feature_names=X.columns, class_names=model.classes_, filled=True)
    plt.title("Decision Tree")
    plt.show()


def main():
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess(df)
    models = get_models()
    results = train_and_evaluate(models, X_train, X_test, y_train, y_test)
    plot_comparison(results)

    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]
    print("\nBest Model:", best_model_name)

    plot_confusion(best_model, X_test, y_test)
    plot_decision_tree(models["Decision Tree"], X_train)


if __name__ == "__main__":
    main()