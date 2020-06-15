from abc import ABC, abstractmethod
from typing import Dict
import pandas as pd
import os

# defines abstract writer class, this allows for different writers
class Abstractwriter(ABC):
    def __init__(self, path: str = 'test'):
        self.path = path

    @abstractmethod
    def write_scalar(self, name: str, value: float) -> None:
        pass

    @abstractmethod
    def save(self):
        pass

    def write_scalars(self, values: Dict[str, float]) -> None:
        for key, value in values.items():
            self.write_scalar(key, value)


# simple writer function, prints key and value
class Printwriter(Abstractwriter):
    def write_scalar(self, name: str, value: float):
        print(name, value)

    def save(self):
        print("k thanks bye")

class CSVwriter(Abstractwriter):
    def __init__(self, path: str = "test.csv"):
        super().__init__(path)
        self.results = {}

    def write_scalar(self, name: str, value: float):
        if name in self.results:
            self.results[name].append(value)
        else:
            self.results[name] = [value]

    def write_list(self, result: list):
        self.results = result

    def save(self):
        # we check if path exists, if not make path
        if not os.path.exists(os.path.dirname(self.path)):
            os.makedirs(os.path.dirname(self.path))
        pd.DataFrame.from_dict(self.results).to_csv(self.path, sep=';')
