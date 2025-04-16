from typing import List, Optional
import strawberry
from enum import Enum

@strawberry.enum
class RecommendationReason(Enum):
    PREVIOUS_PURCHASE = "PREVIOUS_PURCHASE"
    SIMILAR_USERS = "SIMILAR_USERS"
    CATEGORY_AFFINITY = "CATEGORY_AFFINITY"
    BRAND_PREFERENCE = "BRAND_PREFERENCE"
    SEASONAL_MATCH = "SEASONAL_MATCH"

@strawberry.type
class Product:
    ean: str
    name: str
    brand_name: Optional[str]
    category_name: Optional[str]

@strawberry.type
class ProductRecommendation:
    product: Product
    score: float
    reason: RecommendationReason

@strawberry.type
class User:
    id: str

@strawberry.type
class Query:
    @strawberry.field
    async def recommendations(self, user_id: str) -> List[ProductRecommendation]:
        # Implementação virá depois
        pass

schema = strawberry.federation.Schema(
    query=Query,
    enable_federation_2=True,
)    