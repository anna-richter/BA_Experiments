# this function actually runs the whole thing i.e. "start-button"
from experimentor import Experimentor
from writers import Printwriter, Abstractwriter, CSVwriter
import datasets
import json
import argparse
import pandas as pd
import models
import writers
import cleaning
from cleaning import Cleaner
import itertools
import formatting
from formatting import Formatter
from time import time

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
    results = {"data": [],"cleaning": [],"model": [],"layers": [], "runtime": [],
               "nr_of_words_excluded":[], "top_5": []}
    config_path = './config/experimente.json'
    config = json.load(open(config_path, 'r'))

    for data_name, model_name, cleaning in itertools.product(config["data"], config["model"],
                                                             config["cleaning"]):
        save_path = f"./{data}/{model}/"
        for item in cleaning:
            save_path += '_' + item.__name__
        if model == "BERT":
            for layer in config["layers"]:
                save_path += f"/{layer}.csv"
                print("running configuration: ", save_path)
                start = time()
                writer = CSVwriter(save_path)
                data = get_data(data_name)
                model = get_model(model_name)
                cleaner = Cleaner(data_name, data, get_cleaners(cleaning), model, model_name)
                cleandata = cleaner.clean_data()
                experimentor = Experimentor(model, model_name, cleandata, data_name, writer, layer)
                experimentor.run_experiment()
                end = time()
                runtime = end-start

        else:
            save_path += ".csv"
            print("running configuration: ", save_path)
            start = time()
            layer = None
            writer = CSVwriter(save_path)
            data = get_data(data_name)
            model = get_model(model_name)
            cleaner = Cleaner(data_name, data, get_cleaners(cleaning), model, model_name)
            cleandata = cleaner.clean_data()
            experimentor = Experimentor(model, model_name, cleandata, data_name, writer, layer)
            experimentor.run_experiment()
            end = time()
            runtime = end - start