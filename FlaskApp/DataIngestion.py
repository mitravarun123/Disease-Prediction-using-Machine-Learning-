import pandas as pd 
import os 

class DataIngestion:
    def __init__(self):
        base_path = os.path.dirname(__file__)
        csv_path = os.path.join(base_path, 'Training.csv')
        self.data = pd.read_csv(csv_path)
    def get_symptoms(self):
        self.symptoms = [value for value in list(self.data.columns) if value not in ['Unnamed: 133','prognosis']]
        return self.symptoms 
