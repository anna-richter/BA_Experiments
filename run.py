# this function actually runs the whole thing i.e. "start-button"
from experimentor import Experimentor
from writers import Printwriter, Abstractwriter
import datasets
import json
import argparse
import pandas as pd
import models
import writers
import cleaning
from cleaning import Cleaner
import formatting
from formatting import Formatter

# converts string (name of writer) to actual writer object
def get_writer(writer_name: str, **kwargs) -> Abstractwriter:
    return getattr(writers, writer_name)

def get_data(data_name: str, **kwargs) -> pd.DataFrame:
    return getattr(datasets, data_name)()

def get_model(model_name: str, **kwargs) -> object:
    return getattr(models, model_name)()

def get_cleaners(cleaner_names: list) -> callable:
    return [getattr(cleaning, name) for name in cleaner_names]


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
    data = get_data(config["data"])
    model = get_model(config["model"])
    cleaner = Cleaner(config["data"], data, get_cleaners(config["cleaning"]), config["model"])
    cleandata = cleaner.clean_data()

    formatter = Formatter(config["model"], cleandata)
    formatted_data = formatter.format_data()
    experimentor = Experimentor(model, config["model"], formatted_data, config["data"], writer)
    experimentor.run_experiment()
    #print(cleandata)