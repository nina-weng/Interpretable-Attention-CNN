import sys
import time
import logging
from config import config, create_folder
from utils import IOHelper
from experiment import experiment
import torch

class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately
    def flush(self) :
        for f in self.files:
            f.flush()

"""
Main entry of the program
Creates the logging files, loads the data and starts the benchmark.
All configurations (parameters) of this benchmark are specified in config.py
"""

def main():
    torch.backends.cudnn.enabled = False

    isbenchmark = False

    # Setting up logging
    create_folder()
    logging.basicConfig(filename=config['info_log'], level=logging.INFO)
    logging.info('Started the Logging')
    logging.info(f"Using {config['framework']}")
    start_time = time.time()

    # For being able to see progress that some methods use verbose (for debugging purposes)
    f = open(config['model_dir'] + '/console.out', 'w')
    original = sys.stdout
    sys.stdout = Tee(sys.stdout, f)

    #Load the data
    trainX, trainY = IOHelper.get_npz_data(config['data_dir'], verbose=True)


    print('Starting the experiment...')
    logging.info('-' * 30)
    logging.info('Starting the experiment...')

    experiment(trainX,trainY)
    print('Finish the experiment.')
    logging.info('Finish the experiment.')
    logging.info('-' * 30)


    logging.info("--- Runtime: %s seconds ---" % (time.time() - start_time))
    logging.info('Finished Logging')

if __name__=='__main__':
    main()
