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
    results = {"data": [], "model": [], "cleaning": [],"layers": [], "runtime": [], "top_5": []}
    config_path = './config/experimente.json'
    config = json.load(open(config_path, 'r'))

    for i, (data_name, model_name, cleaning_items) in enumerate(itertools.product(config["data"], config["model"],
                                                             config["cleaning"])):
        print('Conducting experiment', i, 'of ', len(config["data"]) * len(config["model"]) * len(config["cleaning"]))

        cleaning_name = "basic"
        for item in cleaning_items:
            if item != "basic":
                cleaning_name += '_' + item
        if model_name == "BERT":
            for layer in config["layers"]:
                save_path = f"./{data_name}/{model_name}/{cleaning_name}"
                save_path += f"/{layer}.csv"
                print("running configuration: ", save_path)
                start = time()
                writer = CSVwriter(save_path)
                data = get_data(data_name)
                model = get_model(model_name)
                cleaner = Cleaner(data_name, data, get_cleaners(cleaning_items), model, model_name)
                cleandata = cleaner.clean_data()
                experimentor = Experimentor(model, model_name, cleandata, data_name, writer, layer)
                titles = experimentor.run_experiment()
                end = time()
                runtime = end-start

                results["data"].append(data_name)
                results["model"].append(model_name)
                results["cleaning"].append(cleaning_name)
                results["layers"].append(layer)
                results["runtime"].append(runtime)
                results["top_5"].append(titles)
                pd.DataFrame.from_dict(results).to_csv("results.csv", sep=";")

        else:
            save_path = f"./{data_name}/{model_name}/{cleaning_name}.csv"
            print("running configuration: ", save_path)
            start = time()
            layer = None
            writer = CSVwriter(save_path)
            data = get_data(data_name)
            model = get_model(model_name)
            cleaner = Cleaner(data_name, data, get_cleaners(cleaning_items), model, model_name)
            cleandata = cleaner.clean_data()
            experimentor = Experimentor(model, model_name, cleandata, data_name, writer, layer)
            titles = experimentor.run_experiment()
            end = time()
            runtime = end - start
            results["data"].append(data_name)
            results["model"].append(model_name)
            results["cleaning"].append(cleaning_name)
            results["layers"].append(layer)
            results["runtime"].append(runtime)
            results["top_5"].append(titles)
            pd.DataFrame.from_dict(results).to_csv("results.csv", sep=";")

