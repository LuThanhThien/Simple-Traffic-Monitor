import glob
import logging
import os
from datetime import datetime

def manage_log_files(logs_path, max_files=10):
    # Get a list of all log files
    log_files = glob.glob(os.path.join(logs_path, "*.log"))

    # If there are more than max_files log files, delete the oldest ones
    if len(log_files) > max_files:
        sorted_files = sorted(log_files, key=os.path.getctime)
        for file in sorted_files[:-max_files]:
            os.remove(file)

script_dir = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(script_dir, "components", "logs")
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

logging.basicConfig(
    filename = LOG_FILE_PATH,
    format = "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level = logging.INFO, 
)   

# Logger files management
manage_log_files(logs_path)

# testing
# if __name__=="__main__":
#     logging.info("Logging has started")
