
import argparse
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="Iris.csv")
    ap.add_argument("--target", default="Species")
    ap.add_argument("--out", default="model.joblib")
    ap.add_argument("--n-estimators", type=int, default=200)
    ap.add_argument("--max-depth", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    df = pd.read_csv(a.data)
    df = df.drop(columns=[c for c in df.columns if c.lower() == "id"])
    X, y = df.drop(columns=[a.target]), df[a.target]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=a.seed, stratify=y
    )

    num = X.select_dtypes("number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]), cat),
    ])
    params = {"n_estimators": a.n_estimators, "max_depth": a.max_depth,
              "random_state": a.seed}
    pipe = Pipeline([("pre", pre), ("clf", RandomForestClassifier(**params))])

    # ---- MLflow: эксперимент, параметры, метрики, модель ----
    mlflow.set_tracking_uri("sqlite:///mlflow.db")      # локально; для сервера: http://host:5000
    mlflow.set_experiment("kaggle-classification")
    with mlflow.start_run(run_name="random_forest"):
        mlflow.log_params(params)
        mlflow.log_param("dataset", a.data)

        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        mlflow.log_metric("accuracy", accuracy_score(y_te, pred))
        mlflow.log_metric("f1_macro", f1_score(y_te, pred, average="macro"))

        sig = infer_signature(X_te, pred)
        mlflow.sklearn.log_model(pipe, "model", signature=sig, skops_trusted_types=['numpy.dtype','sklearn.tree._tree.Tree'],
                                 input_example=X_te.head(3))
        mlflow.log_artifact(a.data) if len(df) < 50000 else None

    joblib.dump(pipe, a.out)
    joblib.dump(list(X.columns), "features.joblib")
    print("accuracy:", accuracy_score(y_te, pred), "-> saved", a.out)


if __name__ == "__main__":
    main()
