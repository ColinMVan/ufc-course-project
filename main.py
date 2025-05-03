# necessary imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


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
        "RedDecOdds",
        "BlueDecOdds",
        "RSubOdds",
        "BSubOdds",
        "RKOOdds",
        "BKOOdds",
        "TotalFightTimeSecs",
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
df_model["DecOddsDiff"] = df_model["RedDecOdds"] - df_model["BlueDecOdds"]
df_model["SubOddsDiff"] = df_model["RSubOdds"] - df_model["BSubOdds"]
df_model["KOOddsDiff"] = df_model["RKOOdds"] - df_model["BKOOdds"]

# drop the old columns
df_model.drop(
    columns=[
        "RedOdds",
        "BlueOdds",
        "RedExpectedValue",
        "BlueExpectedValue",
        "RedDecOdds",
        "BlueDecOdds",
        "RSubOdds",
        "BSubOdds",
        "RKOOdds",
        "BKOOdds",
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
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)

# evaluate decision tree and print results
print("\nDecision Tree:")
print("Accuracy:", accuracy_score(y_test, dt_preds))
