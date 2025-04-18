import time
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt
from rich.table import Table

from src.models.product_similarity import ProductSimilarityEngine
from src.models.recommendation_engine import RecommendationEngine

console = Console()


def create_header() -> Panel:
    """Cria um cabeçalho bonito para a aplicação."""
    return Panel(
        "[bold cyan]Sistema de Recomendações de Produtos[/bold cyan]\n"
        "[blue]Powered by AI[/blue]",
        style="bold white",
    )


def show_loading_animation(message: str):
    """Mostra uma animação de loading com o Rich."""
    with console.status(f"[bold blue]{message}...", spinner="dots"):
        time.sleep(1)


def get_sample_products(engine: RecommendationEngine, n: int = 5) -> Table:
    """Mostra uma amostra dos produtos disponíveis."""
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("EAN")
    table.add_column("Nome")
    table.add_column("Categoria")
    table.add_column("Marca")

    # Pegar os primeiros n produtos como exemplo
    sample_products = list(engine.product_mapping.values())[:n]

    for product in sample_products:
        table.add_row(
            product["ean"],
            product["name"],
            product.get("categoryName", "N/A"),
            product.get("brandName", "N/A"),
        )

    return table


def show_menu() -> str:
    """Mostra o menu principal e retorna a escolha do usuário."""
    choices = {
        "1": "Recomendações por Produto (EAN)",
        "2": "Recomendações por Texto",
        "3": "Ver Produtos de Exemplo",
        "4": "Informações do Sistema",
        "5": "Teste de Recomendações Personalizadas",
        "0": "Sair",
    }

    console.print("\n[bold cyan]Menu Principal[/bold cyan]")
    for key, value in choices.items():
        console.print(f"[blue]{key}[/blue]: {value}")

    return Prompt.ask(
        "\n[bold cyan]Escolha uma opção[/bold cyan]", choices=list(choices.keys())
    )


def get_filters() -> Optional[dict]:
    """Permite ao usuário definir filtros para as recomendações."""
    use_filters = Prompt.ask(
        "\nDeseja aplicar filtros?", choices=["s", "n"], default="n"
    )

    if use_filters.lower() == "n":
        return None

    filters = {}

    category = Prompt.ask("\nFiltrar por categoria? (deixe vazio para pular)")
    if category:
        filters["category"] = category

    brand = Prompt.ask("\nFiltrar por marca? (deixe vazio para pular)")
    if brand:
        filters["brand"] = brand

    return filters if filters else None


def main():
    console.clear()
    console.print(create_header())

    try:
        # Inicializar o engine
        show_loading_animation("Inicializando o motor de recomendações")
        engine = ProductSimilarityEngine()
        engine.load_embeddings(load_format="faiss")
        console.print("[green]✓ Sistema inicializado com sucesso![/green]")

        while True:
            choice = show_menu()

            if choice == "0":
                console.print("\n[bold cyan]Encerrando o sistema...[/bold cyan]")
                break

            elif choice == "1":
                console.print("\n[bold cyan]Recomendações por Produto[/bold cyan]")
                ean = Prompt.ask("\nDigite o EAN do produto")
                k = IntPrompt.ask("Número de recomendações", default=5)

                filters = get_filters()

                show_loading_animation("Buscando recomendações")
                try:
                    recommendations = engine.get_product_recommendations(
                        ean, k=k, filters=filters
                    )
                    console.print(
                        "\n[bold green]Recomendações encontradas:[/bold green]"
                    )
                    engine.print_recommendations(recommendations)
                except ValueError as e:
                    console.print(f"\n[bold red]Erro: {str(e)}[/bold red]")

            elif choice == "2":
                console.print("\n[bold cyan]Recomendações por Texto[/bold cyan]")
                query = Prompt.ask("\nDigite sua busca")
                k = IntPrompt.ask("Número de recomendações", default=5)

                filters = get_filters()

                show_loading_animation("Buscando produtos similares")
                recommendations = engine.get_recommendations_by_text(
                    query, k=k, filters=filters
                )
                console.print("\n[bold green]Produtos encontrados:[/bold green]")
                engine.print_recommendations(recommendations)

            elif choice == "3":
                console.print("\n[bold cyan]Produtos de Exemplo[/bold cyan]")
                n = IntPrompt.ask("Número de produtos para mostrar", default=5)
                table = get_sample_products(engine, n)
                console.print(table)

            elif choice == "4":
                console.print("\n[bold cyan]Informações do Sistema[/bold cyan]")
                metadata = engine.metadata

                info_table = Table(show_header=True, header_style="bold magenta")
                info_table.add_column("Propriedade")
                info_table.add_column("Valor")

                for key, value in metadata.items():
                    info_table.add_row(str(key), str(value))

                console.print(info_table)

            # Adicionar ao menu principal
            elif choice == "5":
                console.print(
                    "\n[bold cyan]Teste de Recomendações Personalizadas[/bold cyan]"
                )
                test_user_recommendations()

            # Pausa antes de mostrar o menu novamente
            Prompt.ask("\nPressione ENTER para continuar")
            console.clear()
            console.print(create_header())

    except Exception as e:
        console.print(f"\n[bold red]Erro fatal: {str(e)}[/bold red]")
        raise

    console.print("\n[bold green]Sistema encerrado com sucesso![/bold green]")


if __name__ == "__main__":
    main()
