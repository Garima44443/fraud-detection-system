import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

def train_models(df):

    if "Class" not in df.columns:
        raise Exception("Dataset must have 'Class' column")

    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 🔥 FAST Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=50,
        n_jobs=-1,
        random_state=42
    )
    rf_model.fit(X_train, y_train)

    # 🔥 FAST XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="logloss",
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)

    return rf_model, xgb_model


def ensemble_predict(rf_model, xgb_model, df, threshold=0.5):

    if "Class" in df.columns:
        df = df.drop("Class", axis=1)

    rf_prob = rf_model.predict_proba(df)[:, 1]
    xgb_prob = xgb_model.predict_proba(df)[:, 1]

    final_pred = []
    final_prob = []

    for i in range(len(rf_prob)):
        prob = (rf_prob[i] + xgb_prob[i]) / 2
        final_prob.append(round(prob, 3))

        if prob > threshold:
            final_pred.append("Fraud")
        else:
            final_pred.append("Normal")

    return final_pred, final_prob


def risk_label(prob):
    if prob > 0.8:
        return "High Risk"
    elif prob > 0.5:
        return "Medium Risk"
    else:
        return "Low Risk"