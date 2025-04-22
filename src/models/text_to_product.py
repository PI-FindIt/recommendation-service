from gliner import GLiNER
from rich.console import Console

from src.api.schema import Product
from src.data.data_service import DataService
from src.models.product_similarity import ProductSimilarityEngine

console = Console()


class TextToProductEngine:
    def __init__(self, similarity_engine: ProductSimilarityEngine) -> None:
        self.model_name = "empathyai/gliner_large-v2.5-groceries"
        self.model = GLiNER.from_pretrained(self.model_name)
        self.data_service = DataService()
        self.similarity_engine = similarity_engine

        self.labels = [
            "Fruits Vegetables",
            "Lactose, Diary, Eggs, Cheese, Yoghurt",
            "Meat, Fish, Seafood",
            "Frozen, Prepared Meals",
            "Baking, Cooking",
            "Cereals, Grains, Canned, Seeds",
            "Breads",
            "Snacks, Pastries, Treats",
            "Frozen Desserts",
            "Hot Drinks, Chilled Drinks",
            "Alcoholic Drinks",
            "Spices, Sauces",
            "World Foods",
            "Dietary Restrictions, Health, Allergens, Lifestyle",
        ]

    def predict(self, text: str) -> list[Product]:
        predictions = self.model.predict_entities(text, labels=self.labels)
        return [
            Product(
                self.similarity_engine.get_recommendations_by_text(
                    product.get("text"), k=1
                )[0].get("ean")
            )
            for product in predictions
        ]
