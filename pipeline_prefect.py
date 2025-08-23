from prefect import flow, task
from datetime import timedelta
import subprocess
import sys
import logging
import src.utils.log_config as log_config


STEPS = [
    ("INGESTION", "src.ingestion.ingest_data"),
    ("VALIDATION", "src.validation.validate_data"),
    ("PREPARATION", "src.preparation.prepare_data"),
    ("TRANSFORMATION_AND_STORAGE", "src.transformation_and_storage.transform_and_store_data"),
    ("MODEL_BUILDING", "src.model_building.model_building")
]


@task
def run_step(task_name, script):
    try:
        logging.info(f"Running step: {task_name} ({script})")
        result = subprocess.run(
            ["python", "-m", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=True
        )
        if result.stdout:
            logging.info(result.stdout)
        if result.stderr:
            logging.warning(result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Step {task_name} ({script}) failed with error: \n{e.stderr}")
        sys.exit(1)


@flow(name="Customer Churn Prefect Pipeline")
def churn_pipeline():
    logging.info('----------------------------INITIATING PIPELINE (PREFECT)----------------------------')
    for step_no, (task_name, script) in enumerate(STEPS, start=1):
        logging.info(f"STEP {step_no}/{len(STEPS)} - {task_name}")
        run_step(task_name, script)
    logging.info('--------------------------PIPELINE RUN SUCCESSFUL (PREFECT)--------------------------')


if __name__ == "__main__":
    churn_pipeline()

    churn_pipeline.serve(
        name="daily-customer-churn",
        cron="*/3 * * * *",
    )
