import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import os
import numpy as np
from sklearn.model_selection import learning_curve


def load_and_preprocess_data(csv_path="ufc-master.csv"):
    """
    Loads the UFC data, performs initial cleanup, and encodes the target variable.
    Handles NaN values explicitly.
    """

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Error: CSV file not found at {csv_path}")

    df = pd.read_csv(csv_path)

    # Data validation
    required_cols = ['Winner', 'RedOdds', 'BlueOdds', 'RedExpectedValue', 'BlueExpectedValue']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Error: Column '{col}' not found in CSV")
        if col != 'Winner' and df[col].dtype not in ['int64', 'float64']:
            try:
                df[col] = pd.to_numeric(df[col])
            except ValueError:
                raise ValueError(f"Error: Column '{col}' must be numeric")

    df = df.dropna(subset=['Winner'])
    df = df[df['Winner'].isin(['Red', 'Blue'])]

    # Explicit NaN handling: Impute
    for col in ['RedOdds', 'BlueOdds', 'RedExpectedValue', 'BlueExpectedValue']:
        if df[col].isnull().any():
            print(f"Column '{col}' has NaN values. Imputing with the mean.")
            df[col] = df[col].fillna(df[col].mean())

    df['WinnerEncoded'] = df['Winner'].map({'Red': 1, 'Blue': 0})
    return df

def engineer_features(df):
    """
    Engineers features like OddsDiff and ExpectedValueDiff.
    Handles potential division by zero and NaN propagation.
    """
    df['OddsDiff'] = df['RedOdds'] - df['BlueOdds']
    # Use np.where to avoid division by zero and handle potential NaN propagation
    df['ExpectedValueDiff'] = np.where(
        (df['RedExpectedValue'].notna()) & (df['BlueExpectedValue'].notna()) &
        (df['RedExpectedValue'] != 0) & (df['BlueExpectedValue'] != 0),
        df['RedExpectedValue'] - df['BlueExpectedValue'],
        0  # Or np.nan if you want to explicitly mark these cases
    )
    return df


def train_logistic_regression(X_train_scaled, y_train):
    """
    Trains a Logistic Regression model with hyperparameter tuning.
    Returns both the best model and best cross-validated AUC score.
    """
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
    grid_search = GridSearchCV(
        LogisticRegression(max_iter=1000, solver='liblinear'),
        param_grid,
        cv=5,
        scoring='roc_auc',
        return_train_score=True
    )
    grid_search.fit(X_train_scaled, y_train)
    best_auc = grid_search.best_score_
    print(f"Best Logistic Regression parameters: {grid_search.best_params_}")
    print(f"Best Logistic Regression cross-validated ROC AUC: {best_auc:.4f}")
    return grid_search.best_estimator_, best_auc



def train_decision_tree(X_train, y_train):
    """
    Trains a Decision Tree model with hyperparameter tuning.
    Returns both the best model and best cross-validated AUC score.
    """
    param_grid = {
        'max_depth': [3, 5, 7],
        'min_samples_split': [5, 10, 20],
        'min_samples_leaf': [3, 5, 10]
    }
    grid_search = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring='roc_auc',
        return_train_score=True
    )
    grid_search.fit(X_train, y_train)
    best_auc = grid_search.best_score_
    print(f"Best Decision Tree parameters: {grid_search.best_params_}")
    print(f"Best Decision Tree cross-validated ROC AUC: {best_auc:.4f}")
    return grid_search.best_estimator_, best_auc




def evaluate_model(model, X_test_scaled, y_test, model_name):
    """
    Evaluates a model and calculates ROC AUC.
    """

    y_probs = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
    print(f"{model_name} Accuracy: {accuracy:.4f}")
    return fpr, tpr, roc_auc

def plot_learning_curve(estimator, X, y, title, scaler=None, filename="learning_curve.png"):
    """
    Plots a learning curve for a given estimator.
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, scoring='accuracy', n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5), shuffle=True, random_state=42
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    fig = plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label="Cross-validation score")
    plt.title(f"Learning Curve: {title}")
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.grid()
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved learning curve to {filename}")
    return fig

def plot_results(df, log_reg, tree, X_test_scaled, y_test, fpr_log, tpr_log, roc_auc_log, fpr_tree, tpr_tree, roc_auc_tree):
    """
    Generates and saves the visualizations.
    """

    figs = []

    # Sample Statistics Table
    stats_df = df[['RedOdds', 'BlueOdds', 'RedExpectedValue', 'BlueExpectedValue']].describe()
    print("\nSample Statistics for Odds and Expected Value:\n")
    print(stats_df)

    # Histograms
    fig_hist, axes_hist = plt.subplots(2, 2, figsize=(10, 8))
    df['RedOdds'].hist(ax=axes_hist[0, 0], bins=20, color='red', alpha=0.7)
    axes_hist[0, 0].set_title('Red Odds Distribution')
    df['BlueOdds'].hist(ax=axes_hist[0, 1], bins=20, color='blue', alpha=0.7)
    axes_hist[0, 1].set_title('Blue Odds Distribution')
    df['RedExpectedValue'].hist(ax=axes_hist[1, 0], bins=20, color='red', alpha=0.7)
    axes_hist[1, 0].set_title('Red Expected Value Distribution')
    df['BlueExpectedValue'].hist(ax=axes_hist[1, 1], bins=20, color='blue', alpha=0.7)
    axes_hist[1, 1].set_title('Blue Expected Value Distribution')
    plt.tight_layout()
    plt.savefig("odds_expected_value_histograms.png")
    figs.append(fig_hist)


    fig1 = plt.figure(figsize=(6, 4))
    df['Winner'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['red', 'blue'])
    plt.title("Winner Distribution")
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig("winner_distribution_pie.png")
    figs.append(fig1)

    fig2 = plt.figure(figsize=(6, 4))
    sns.boxplot(data=df[['OddsDiff', 'ExpectedValueDiff']])
    plt.title("Boxplot of Engineered Features")
    plt.tight_layout()
    plt.savefig("boxplot_features.png")
    figs.append(fig2)

    fig3 = plt.figure(figsize=(6, 4))
    plt.bar(['Logistic Regression', 'Decision Tree'],
            [log_reg.score(X_test_scaled, y_test), tree.score(X_test_scaled, y_test)],
            color=['green', 'purple'])
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("model_accuracy_comparison.png")
    figs.append(fig3)

    fig4 = plt.figure(figsize=(6, 4))
    plt.plot(fpr_log, tpr_log, color='green', label=f'Logistic Regression (AUC = {roc_auc_log:.2f})')
    plt.plot(fpr_tree, tpr_tree, color='purple', label=f'Decision Tree (AUC = {roc_auc_tree:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.title('ROC Curve Comparison')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_curve_comparison.png")
    figs.append(fig4)

    fig5, axes = plt.subplots(1, 2, figsize=(10, 4))
    log_disp = ConfusionMatrixDisplay.from_estimator(log_reg, X_test_scaled, y_test, ax=axes[0], display_labels=["Blue", "Red"])
    log_disp.ax_.set_title("Logistic Regression Confusion Matrix")
    tree_disp = ConfusionMatrixDisplay.from_estimator(tree, X_test_scaled, y_test, ax=axes[1], display_labels=["Blue", "Red"])
    tree_disp.ax_.set_title("Decision Tree Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrices.png")
    figs.append(fig5)

    plt.show()


def main():
    """
    Main function to execute the script.
    """

    csv_path = "ufc-master.csv"

    try:
        df = load_and_preprocess_data(csv_path)
        df = engineer_features(df)

        X = df[['OddsDiff', 'ExpectedValueDiff']]
        y = df['WinnerEncoded']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        log_reg, log_cv_auc = train_logistic_regression(X_train_scaled, y_train)
        tree, tree_cv_auc = train_decision_tree(X_train, y_train)


        fpr_log, tpr_log, roc_auc_log = evaluate_model(log_reg, X_test_scaled, y_test, "Logistic Regression")
        fpr_tree, tpr_tree, roc_auc_tree = evaluate_model(tree, X_test_scaled, y_test, "Decision Tree")

        plot_results(df.copy(), log_reg, tree, X_test_scaled.copy(), y_test.copy(),
                    fpr_log.copy(), tpr_log.copy(), roc_auc_log,
                    fpr_tree.copy(), tpr_tree.copy(), roc_auc_tree)
        # Learning Curve: Logistic Regression
        plot_learning_curve(log_reg, X_train_scaled, y_train,
                            title="Logistic Regression",
                            filename="learning_curve_logreg.png")

        # Learning Curve: Decision Tree (no scaling)
        plot_learning_curve(tree, X_train, y_train,
                            title="Decision Tree",
                            filename="learning_curve_tree.png")

        print("Visualizations saved successfully.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()