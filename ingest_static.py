import pandas as pd
from datetime import datetime
import os


def fetch_static_data(source_path):
    df = pd.read_csv(source_path)
    print(df.shape)
    print(df.head())
    today = datetime.today()
    raw_static_file_storage_directory = f'data/raw/static/{today}'
    os.makedirs(raw_static_file_storage_directory, exist_ok=True)
    destination_file = os.path.join(raw_static_file_storage_directory, os.path.basename(source_path))
    df.to_csv(destination_file, index=False)


fetch_static_data('data/Telco-Customer-Churn.csv')
