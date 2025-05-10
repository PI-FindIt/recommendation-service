import sys

from rich.console import Console

from src.api.cli import main_cli
from src.data.data_service import DataService
from src.models.product_similarity import ProductSimilarityEngine
from src.models.text_to_product import TextToProductEngine
from src.models.user_recommendation_engine import UserRecommendationEngine

# if sys.platform == "darwin":
#     import torch
#     torch.set_num_threads(1)

console = Console()

with console.status("[bold blue]Starting engines...", spinner="dots"):
    engine = ProductSimilarityEngine()
    engine.load_embeddings(load_format="faiss")
    user_recommendation_engine = UserRecommendationEngine()
    user_recommendation_engine.data_service = DataService(
        api_url="http://192.168.1.159"
    )
    text_to_product_engine = TextToProductEngine(engine)

console.print("[green]✓ System started successfully![/green]")

if __name__ == "__main__":
    main_cli(engine, text_to_product_engine)
    sys.exit(0)


import strawberry

from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from strawberry.extensions.tracing import OpenTelemetryExtension
from strawberry.fastapi import GraphQLRouter

from src.config import settings
from src.api.schema import Product, RecommendationFilterInput


@strawberry.type
class Query:
    @strawberry.field()
    async def text_to_product(self, text: str) -> list[Product]:
        return [
            Product(ean=product.get("product").get("ean"))
            for product in text_to_product_engine.predict(text)
        ]

    @strawberry.field()
    async def raw_list_by_user_and_ai(self, user_id: str) -> list[Product]:
        return [
            Product(a.get("product").get("ean"))
            for a in user_recommendation_engine.get_recommendations(user_id, k=20)
        ]

    @strawberry.field()
    async def recommendations_by_product(
        self,
        ean: str,
        recommendations: int = 5,
        filters: RecommendationFilterInput | None = None,
    ) -> list[Product]:
        return [
            Product(ean=product.get("product").get("ean"))
            for product in engine.get_product_recommendations(
                ean,
                k=recommendations,
                filters=strawberry.asdict(filters) if filters else None,
            )
        ]

    @strawberry.field()
    async def recommendations_by_text(
        self,
        query: str,
        recommendations: int = 5,
        filters: RecommendationFilterInput | None = None,
    ) -> list[Product]:
        return [
            Product(ean=product.get("product").get("ean"))
            for product in engine.get_recommendations_by_text(
                query,
                k=recommendations,
                filters=strawberry.asdict(filters) if filters else None,
            )
        ]


schema = strawberry.federation.Schema(
    query=Query,
    extensions=[OpenTelemetryExtension] if settings.TELEMETRY else [],
    enable_federation_2=True,
)
graphql_app = GraphQLRouter(schema)

app = FastAPI(title="Recommendation Service")
app.include_router(graphql_app, prefix="/graphql")


@app.get("/ping")
def ping() -> dict[str, str]:
    return {"message": "pong"}


if settings.TELEMETRY:
    resource = Resource(attributes={SERVICE_NAME: "product-service"})
    tracer = TracerProvider(resource=resource)

    otlp_exporter = OTLPSpanExporter(
        endpoint="apm-server:8200",
        insecure=True,
    )
    tracer.add_span_processor(BatchSpanProcessor(otlp_exporter))
    trace.set_tracer_provider(tracer)

    FastAPIInstrumentor.instrument_app(app)
