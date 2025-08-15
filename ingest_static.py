import pandas as pd
from datetime import datetime
import os
import logging
import log_config

def fetch_static_data(source_path):

    try:
        df = pd.read_csv(source_path)
    except Exception as e:
        logging.error(f"Static ingestion failed: {e}", exc_info=True)
        print(f"Error during static ingestion: {e}")

    today = datetime.today()
    raw_static_file_storage_directory = f'data/raw/static/{today}'
    os.makedirs(raw_static_file_storage_directory, exist_ok=True)
    destination_file = os.path.join(raw_static_file_storage_directory, os.path.basename(source_path))

    df.to_csv(destination_file, index=False)

    logging.info(f"Static ingestion successful. Rows: {len(df)}")
    print(f'Static data ingestion complete. Rows: {len(df)}')

    return df