import pandas as pd
import logging
from datetime import datetime
import os

import log_config
from ingest_static import fetch_static_data
from ingest_live import generate_live_data


def main():
    try:
        static_df = fetch_static_data('data/Telco-Customer-Churn.csv')

        live_df = generate_live_data(n=25)

        combined_df = pd.concat([static_df, live_df])
        today = datetime.now().strftime('%Y-%m-%d')
        combined_dir = f'data/raw/combined/{today}'
        os.makedirs(combined_dir, exist_ok=True)
        path_to_save_combined_df = os.path.join(combined_dir, 'combined.csv')
        combined_df.to_csv(path_to_save_combined_df, index=False)

        logging.info(f"Ingestion successful. Rows: {len(combined_df)}")
        print(f'Ingestion complete.Rows: {len(combined_df)}')

    except Exception as e:
        logging.error(f"Ingestion failed: {e}", exc_info=True)
        print(f"Error during ingestion: {e}")

if __name__=='__main__':
    main()