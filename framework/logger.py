from framework.exception import MyException
import os,sys,logging
from datetime import datetime
from framework.fetch_config import GetConfig


script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
# script_name_1 = os.path.splitext(os.path.basename(sys.argv[0]))[1]

log_path = GetConfig(config_file='dir_path.yaml',variables='log_dir').get()

os.makedirs(log_path, exist_ok=True)
LOG_FILE_PATH = os.path.join(log_path, '{0}_{1}.log'.format(script_name,datetime.now().strftime('%m_%d_%Y_%H_%M_%S')))

logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.DEBUG,
    # format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    format = f'%(asctime)s - %(levelname)s - %(message)s - %(name)s - %(lineno)d'
    )






