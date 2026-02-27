import pandas as pd
from src.data.preprocess import preprocess_data
from src.models.train import train_model
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def main():

    # Load dataset
    df = pd.read_excel("data/raw/credit_default_raw.xls", header=1)

    # Rename target column
    df.rename(columns={"default payment next month": "default"}, inplace=True)

    # Drop ID column
    df.drop("ID", axis=1, inplace=True)

    # Preprocess
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)

    # Train
    train_model(X_train, y_train, X_test, y_test, preprocessor)


if __name__ == "__main__":
    main()
