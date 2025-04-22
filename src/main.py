import sys

from rich.console import Console

from src.api.cli import main_cli
from src.models.product_similarity import ProductSimilarityEngine
from src.models.text_to_product import TextToProductEngine

console = Console()

with console.status("[bold blue]Starting engines...", spinner="dots"):
    engine = ProductSimilarityEngine()
    engine.load_embeddings(load_format="faiss")
    text_to_product_engine = TextToProductEngine(engine)

console.print("[green]✓ System started successfully![/green]")

if __name__ == "__main__":
    main_cli(engine, text_to_product_engine)
    sys.exit(0)
