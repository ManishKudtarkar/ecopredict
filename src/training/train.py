import pandas as pd
from src.models.random_forest import RandomForestModel

def train_model(data_path, params, model_path):
    df = pd.read_csv(data_path)

    X = df.drop("risk_label", axis=1)
    y = df["risk_label"]

    model = RandomForestModel(params)
    model.train(X, y)
    model.save(model_path)
