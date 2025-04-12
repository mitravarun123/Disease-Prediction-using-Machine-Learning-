import pickle 
import numpy as np 
import os
from DataIngestion import DataIngestion 

data = DataIngestion()
# Disease label mapping
index_to_disease = {
    0: 'Fungal infection',
    1: 'Allergy',
    2: 'GERD',
    3: 'Chronic cholestasis',
    4: 'Drug Reaction',
    5: 'Peptic ulcer diseae',
    6: 'AIDS',
    7: 'Diabetes',
    8: 'Gastroenteritis',
    9: 'Bronchial Asthma',
    10: 'Hypertension',
    11: 'Migraine',
    12: 'Cervical spondylosis',
    13: 'Paralysis (brain hemorrhage)',
    14: 'Jaundice',
    15: 'Malaria',
    16: 'Chicken pox',
    17: 'Dengue',
    18: 'Typhoid',
    19: 'hepatitis A',
    20: 'Hepatitis B',
    21: 'Hepatitis C',
    22: 'Hepatitis D',
    23: 'Hepatitis E',
    24: 'Alcoholic hepatitis',
    25: 'Tuberculosis',
    26: 'Common Cold',
    27: 'Pneumonia',
    28: 'Dimorphic hemorrhoids (piles)',
    29: 'Heart attack',
    30: 'Varicose veins',
    31: 'Hypothyroidism',
    32: 'Hyperthyroidism',
    33: 'Hypoglycemia',
    34: 'Osteoarthritis',
    35: 'Arthritis',
    36: '(vertigo) Paroxysmal Positional Vertigo',
    37: 'Acne',
    38: 'Urinary tract infection',
    39: 'Psoriasis',
    40: 'Impetigo'
}

class Predictions:
    def __init__(self):
        base_path = os.path.dirname(__file__)
        model_path = os.path.join(base_path, 'NaiveBayes_model.pkl')
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        self.columns = data.get_symptoms()
    def predict(self, input_features):
        input_vector = [1 if col in input_features else 0 for col in self.columns]
        input_vector = np.array(input_vector).reshape(1, -1)
        prediction = self.model.predict(input_vector)
        return str(index_to_disease[prediction[0]])

