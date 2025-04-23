import strawberry
from enum import Enum


@strawberry.federation.type(keys=["ean"], extend=True)
class Product:
    ean: str


@strawberry.federation.type(keys=["id"], extend=True)
class User:
    id: str


@strawberry.input
class RecommendationFilterInput:
    category: str | None = None
    brand: str | None = None


class RecommendationReason(Enum):
    PREVIOUS_PURCHASE = "PREVIOUS_PURCHASE"
    SIMILAR_USERS = "SIMILAR_USERS"
    CATEGORY_AFFINITY = "CATEGORY_AFFINITY"
    BRAND_PREFERENCE = "BRAND_PREFERENCE"
    SEASONAL_MATCH = "SEASONAL_MATCH"


@strawberry.type
class ProductRecommendation:
    product: Product
    score: float
    reason: RecommendationReason
