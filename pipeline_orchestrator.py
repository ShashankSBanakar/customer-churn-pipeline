import subprocess
import logging
import sys
import src.utils.log_config as log_config

STEPS = [
    ("INGESTION", "src.ingestion.ingest_data"),
    ("VALIDATION", "src.validation.validate_data"),
    ("PREPARATION", "src.preparation.prepare_data"),
    ("TRANSFORMATION_AND_STORAGE", "src.transformation_and_storage.transform_and_store_data"),
    ("MODEL_BUILDING", "src.model_building.model_building")
]

def run_step(script):
    try:
        result = subprocess.run(
                ["python", "-m", script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                check=True
            )
        
        if result.stderr:
            logging.warning(result.stderr)
    
    except subprocess.CalledProcessError as e:
        logging.error(f"Step {script} failed with error: \n{e.stderr}")
        sys.exit(1)

def main():
    logging.info('----------------------------INITIATING PIPELINE----------------------------')

    for step_no, (task, script) in enumerate(STEPS):
        logging.info(f"STEP {step_no+1}/{len(STEPS)} - {task} - {script}")
        run_step(script)

    logging.info('--------------------------PIPELINE RUN SUCCESSFUL--------------------------')


if __name__=="__main__":
    main()