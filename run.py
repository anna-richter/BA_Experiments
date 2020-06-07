# this function actually runs the whole thing i.e. "start-button"
from experimentor import Experimentor
from writers import Printwriter, Abstractwriter
import json
import argparse

import writers

# converts string (name of writer) to actual writer object
def get_writer(writer_name: str, **kwargs) -> Abstractwriter:
    return getattr(writers, writer_name)

# here the actual run starts, we need to make sure that this is actually the current "main method"
if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument('-c', dest='config')
    #parser.parse_args()

    config_path = './config/test.json'
    config = json.load(open(config_path, 'r'))
    print("hello world", config)

    # get the writer class from the name in config
    writer_class = get_writer(config['writer'])
    # make an instance of this writer with the path given in config
    writer = writer_class(config['save_path'])
    experimentor = Experimentor(writer)
    experimentor.run_experiment()