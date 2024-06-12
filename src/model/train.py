# Import libraries
import os
import glob
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import mlflow


# define functions
def main(args):
    # TO DO: enable autologging
    mlflow.autolog()

    # read data
    df = get_csvs_df(args.training_data)

    # split data
    X_train, X_test, y_train, y_test = split_data(df)

    # train model
    train_model(args.reg_rate, X_train, X_test, y_train, y_test)


def get_csvs_df(path):
    if not os.path.exists(path):
        raise RuntimeError(f"Cannot use non-existent path provided: {path}")
    csv_files = glob.glob(f"{path}/*.csv")
    if not csv_files:
        raise RuntimeError(f"No CSV files found in provided data path: {path}")
    return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)


# TO DO: add function to split data
def split_data(df, test_size=0.2, random_state=42):
    x=df.drop(columns=['Diabetic'])
    y=df['Diabetic']

    X_train,X_test,y_test,y_train=train_test_split(X,y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
              

def train_model(reg_rate, X_train, X_test, y_train, y_test):
    # train model
    model=LogisticRegression(C=1/reg_rate, solver="liblinear")
    model.fit(X_train, y_train)
    y_pred=model.predict(X_test)
    
   #log metrics
   accuracy=accuracy_score(y_test,y_pred)
   mlflow.log_metric("accuracy", accuracy)
   mlflow.log_param("reg_rate", reg_rate)

def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--training_data", dest='training_data',
                        type=str)
    parser.add_argument("--reg_rate", dest='reg_rate',
                        type=float, default=0.01)

    # return args
    return parser.parse_args()

# run script
if __name__ == "__main__":

    # parse args
    args = parse_args()

with mlflow.start_run():
    mlflow.log_param("training_data", args.training_data)
    mlflow.log_param("reg_rate", args.reg_rate)

    # run main function
    main(args)

    # add space in logs
    print("*" * 60)
    print("\n\n")
