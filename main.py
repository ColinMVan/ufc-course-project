# necessary imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# PRE-PROCESSING PHASE

# read dataset
df = pd.read_csv("ufc-master.csv")

# copy desired featueres to new data frame
df_model = df[
    [
        "RedOdds",
        "BlueOdds",
        "RedExpectedValue",
        "BlueExpectedValue",
        "Winner",
    ]
].copy()

# drop rows that dont have the winner feature
df_model.dropna(subset=["Winner"], inplace=True)

# create new features
df_model["OddsDiff"] = df_model["RedOdds"] - df_model["BlueOdds"]
df_model["ExpectedValueDiff"] = (
    df_model["RedExpectedValue"] - df_model["BlueExpectedValue"]
)

# drop the old columns
df_model.drop(
    columns=[
        "RedOdds",
        "BlueOdds",
        "RedExpectedValue",
        "BlueExpectedValue",
    ],
    inplace=True,
)

# encode the target variable
df_model["Winner"] = df_model["Winner"].map({"Red": 1, "Blue": 0})

# drop any missing values, last step of preprocessing the data
df_model.dropna(inplace=True)


# TRAINING/TESTING AND EVALUATING PHASE

# split data into training and testing sets
X = df_model.drop(columns=["Winner"])
y = df_model["Winner"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# scale features using standardization, for logistic regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# train logistic regression
logreg_model = LogisticRegression(max_iter=1000)
logreg_model.fit(X_train_scaled, y_train)
logreg_preds = logreg_model.predict(X_test_scaled)

# evaluate logistic regression and print results
print("Logistic Regression:")
print("Accuracy:", accuracy_score(y_test, logreg_preds))

# train decision tree
dt_model = DecisionTreeClassifier(
    random_state=42, max_depth=5, min_samples_split=10, min_samples_leaf=5
)
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)


# evaluate decision tree and print results
print("\nDecision Tree:")
print("Accuracy:", accuracy_score(y_test, dt_preds))

# imports for the plotting and ROC curve
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# get the predicted probabilities for each instance belonging to the positive class
logreg_probs = logreg_model.predict_proba(X_test_scaled)[:, 1]
dt_probs = dt_model.predict_proba(X_test)[:, 1]

# calculate the false/true positive rates
logreg_fpr, logreg_tpr, _ = roc_curve(y_test, logreg_probs)
dt_fpr, dt_tpr, _ = roc_curve(y_test, dt_probs)

# calculate the AUC's for both
logreg_auc = auc(logreg_fpr, logreg_tpr)
dt_auc = auc(dt_fpr, dt_tpr)

# plot both of the ROC curves
plt.figure()
plt.plot(logreg_fpr, logreg_tpr, label=f"Logistic Regression (AUC = {logreg_auc:.2f})")
plt.plot(dt_fpr, dt_tpr, label=f"Decision Tree (AUC = {dt_auc:.2f})")
plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# necessary imports for the cross validation scores
from sklearn.model_selection import cross_val_score

# gets the cross validation scores for both models and prints them
logreg_cv_scores = cross_val_score(
    logreg_model,
    X_train_scaled,
    y_train,
    cv=5,
    scoring="roc_auc",
)
print("Logistic Regression CV AUC Mean:", logreg_cv_scores.mean())
dt_cv_scores = cross_val_score(dt_model, X_train, y_train, cv=5, scoring="roc_auc")
print("Decision Tree CV AUC Mean:", dt_cv_scores.mean())
