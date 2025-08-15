import os
import logging

# set up logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/ingestion.log",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S"
)

logging.info('--------------Initiating flow--------------')