import json
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def generate_product_embeddings(keywords):
    return model.encode(" ".join(keywords)).tolist()

def transform_nutrition(nutrition_str):
    try:
        data = json.loads(nutrition_str)
        return {
            'calories': data.get('energyKcal', 0),
            'proteins': data.get('proteins', 0),
            'carbs': data.get('carbohydrates', 0)
        }
    except:
        return {'calories': 0, 'proteins': 0, 'carbs': 0}

def calculate_price_stats(prices):
    if not prices:
        return {'min': 0, 'max': 0, 'avg': 0}
    return {
        'min': min(prices),
        'max': max(prices),
        'avg': sum(prices)/len(prices)
    }