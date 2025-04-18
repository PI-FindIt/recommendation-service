import json
from typing import List, Dict, Any, Optional

import numpy as np
import requests
from rich.console import Console
from rich.table import Table

from src.models.base_engine import BaseEngine
from src.utils.query import PRODUCTS_QUERY

console = Console()


class ProductSimilarityEngine(BaseEngine):
    def __init__(self):
        field_weights = {
            "name": 0.35,  # Nome do produto é muito importante
            "generic_name": 0.15,  # Nome genérico tem relevância média
            "category": 0.20,  # Categoria é importante para similaridade
            "brand": 0.15,  # Marca tem relevância média
            "keywords": 0.10,  # Keywords ajudam na contextualização
            "ingredients": 0.05,  # Ingredientes têm menor peso mas ainda são relevantes
        }
        super().__init__(field_weights=field_weights, engine_name="product_engine")
        console.print("[bold blue]Inicializando ProductSimilarityEngine...[/bold blue]")

        console.print("\n[bold cyan]Pesos definidos para cada campo:[/bold cyan]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Campo")
        table.add_column("Peso")
        for field, weight in self.field_weights.items():
            table.add_row(field, f"{weight:.2f}")
        console.print(table)

    def _generate_field_embeddings(
        self, product: Dict[str, str]
    ) -> dict[str, np.ndarray]:
        """
        Gera embeddings separados para cada campo do produto.
        """
        embeddings = {}

        # Nome do produto
        if product.get("name"):
            embeddings["name"] = self.model.encode(
                product["name"], convert_to_numpy=True, device=self.device
            )

        # Nome genérico
        if product.get("genericName"):
            embeddings["generic_name"] = self.model.encode(
                product["genericName"], convert_to_numpy=True, device=self.device
            )

        # Categoria
        if product.get("categoryName"):
            embeddings["category"] = self.model.encode(
                product["categoryName"], convert_to_numpy=True, device=self.device
            )

        # Marca
        if product.get("brandName"):
            embeddings["brand"] = self.model.encode(
                product["brandName"], convert_to_numpy=True, device=self.device
            )

        # Keywords
        if product.get("keywords") and len(product["keywords"]) > 0:
            embeddings["keywords"] = self.model.encode(
                " ".join(product["keywords"]), convert_to_numpy=True, device=self.device
            )

        # Ingredientes
        if product.get("ingredients"):
            embeddings["ingredients"] = self.model.encode(
                product["ingredients"], convert_to_numpy=True, device=self.device
            )

        return embeddings

    def find_similar_products(
        self, product_idx: int, k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Encontra os k produtos mais similares ao produto dado.
        """
        if self.index is None:
            raise ValueError("Índice não foi construído ainda!")

        console.print(f"\n[bold cyan]Buscando {k} produtos similares...[/bold cyan]")

        console.print(self.data_mapping, product_idx)
        product = self.data_mapping[product_idx]
        console.print("\n[yellow]Produto de referência:[/yellow]")
        console.print(f"Nome: {product['name']}")
        console.print(f"EAN: {product['ean']}")
        console.print(f"Categoria: {product.get('categoryName', 'N/A')}")
        console.print(f"Marca: {product.get('brandName', 'N/A')}")

        field_embeddings = self._generate_field_embeddings(product)
        query_embedding = self._combine_embeddings(field_embeddings)

        console.print(
            "\n[bold green]Contribuição de cada campo para o embedding final:[/bold green]"
        )
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Campo")
        table.add_column("Peso Efetivo")
        table.add_column("Norma do Embedding")

        for field, emb in field_embeddings.items():
            weight = self.field_weights[field]
            norm = np.linalg.norm(emb)
            table.add_row(field, f"{weight:.3f}", f"{norm:.3f}")
        console.print(table)

        console.print(
            "\n[yellow]Buscando produtos similares no índice FAISS...[/yellow]"
        )
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k + 1)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != product_idx:  # Não incluir o próprio produto
                similar_product = self.data_mapping[idx]
                similarity_score = float(1.0 / (1.0 + dist))
                results.append(
                    {
                        "product": similar_product,
                        "similarity_score": similarity_score,
                        "distance": float(dist),
                    }
                )

        return results

    def get_product_recommendations(
        self, product_ean: str, k: int = 5, filters: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Obtém recomendações para um produto específico.

        Args:
            product_ean: EAN do produto
            k: Número de recomendações
            filters: Filtros opcionais (categoria, marca, etc.)
        """
        # Encontrar o índice do produto
        product_idx = None
        for idx, product in self.data_mapping.items():
            if product["ean"] == product_ean:
                product_idx = idx
                break

        if product_idx is None:
            raise ValueError(f"Produto com EAN {product_ean} não encontrado!")

        # Obter o produto de referência
        product = self.data_mapping[product_idx]
        console.print("\n[bold cyan]Buscando recomendações para:[/bold cyan]")
        console.print(f"Nome: {product['name']}")
        console.print(f"EAN: {product['ean']}")
        console.print(f"Categoria: {product.get('categoryName', 'N/A')}")
        console.print(f"Marca: {product.get('brandName', 'N/A')}")

        # Buscar produtos similares
        query_embedding = self.index.reconstruct(product_idx).reshape(1, -1)
        distances, indices = self.index.search(query_embedding, k + 1)

        # Formatar resultados
        recommendations = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != product_idx:  # Não incluir o próprio produto
                similar_product = self.data_mapping[idx]

                # Aplicar filtros se existirem
                if filters:
                    if (
                        "category" in filters
                        and similar_product.get("categoryName") != filters["category"]
                    ):
                        continue
                    if (
                        "brand" in filters
                        and similar_product.get("brandName") != filters["brand"]
                    ):
                        continue

                recommendations.append(
                    {
                        "product": similar_product,
                        "similarity_score": float(1.0 / (1.0 + dist)),
                        "distance": float(dist),
                    }
                )

        return recommendations[:k]

    def get_recommendations_by_text(
        self, text_query: str, k: int = 5, filters: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Obtém recomendações baseadas em uma query de texto.

        Args:
            text_query: Texto para buscar produtos similares
            k: Número de recomendações
            filters: Filtros opcionais
        """
        console.print(
            f"\n[bold cyan]Buscando produtos similares a: {text_query}[/bold cyan]"
        )

        # Gerar embedding para a query
        query_embedding = self.model.encode(text_query, device=self.device)
        query_embedding = query_embedding.reshape(1, -1).astype("float32")

        # Buscar produtos similares
        distances, indices = self.index.search(query_embedding, k)

        # Formatar resultados
        recommendations = []
        for dist, idx in zip(distances[0], indices[0]):
            product = self.data_mapping[idx]

            # Aplicar filtros
            if filters:
                if (
                    "category" in filters
                    and product.get("categoryName") != filters["category"]
                ):
                    continue
                if "brand" in filters and product.get("brandName") != filters["brand"]:
                    continue

            recommendations.append(
                {
                    "product": product,
                    "similarity_score": float(1.0 / (1.0 + dist)),
                    "distance": float(dist),
                }
            )

        return recommendations

    def print_recommendations(self, recommendations: List[Dict[str, Any]]):
        """
        Imprime as recomendações de forma formatada.
        """
        if not recommendations:
            console.print("[yellow]Nenhuma recomendação encontrada![/yellow]")
            return

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Nome")
        table.add_column("EAN")
        table.add_column("Categoria")
        table.add_column("Marca")
        table.add_column("Score")

        for rec in recommendations:
            product = rec["product"]
            score = rec["similarity_score"]

            table.add_row(
                product["name"],
                product["ean"],
                product.get("categoryName", "N/A"),
                product.get("brandName", "N/A"),
                f"{score:.3f}",
            )

        console.print(table)


def fetch_products(limit: int = 14000) -> List[Dict[str, Any]]:
    """
    Busca produtos da API GraphQL.
    """
    console.print("\n[bold cyan]Iniciando busca de produtos...[/bold cyan]")

    variables = {"filters": {}}

    console.print(
        f"[yellow]Fazendo requisição GraphQL para buscar até {limit} produtos...[/yellow]"
    )

    try:
        try:
            response = requests.post(
                "http://localhost",
                json={"query": PRODUCTS_QUERY, "variables": variables},
            )

            if response.status_code != 200:
                console.print(
                    f"[bold red]Erro na requisição: Status {response.status_code}[/bold red]"
                )
                raise Exception(f"Falha ao buscar produtos: {response.text}")

            data = response.json()
            products = data["data"]["products"][:limit]

        except:
            console.print(
                "[yellow]Não foi possível buscar da API, tentando carregar de products.json...[/yellow]"
            )
            try:
                with open("products.json", "r") as f:
                    products = json.load(f)[:limit]
            except:
                console.print("[bold red]Erro ao carregar products.json[/bold red]")
                raise

        console.print("[green]✓ Busca concluída com sucesso![/green]")
        console.print(f"[blue]Total de produtos encontrados: {len(products)}[/blue]")

        console.print("\n[bold green]Exemplos de produtos encontrados:[/bold green]")
        for i in range(min(3, len(products))):
            p = products[i]
            console.print(f"\n[cyan]Produto {i+1}:[/cyan]")
            console.print(f"Nome: {p['name']}")
            console.print(f"EAN: {p['ean']}")
            console.print(f"Categoria: {p.get('categoryName', 'N/A')}")
            console.print(f"Marca: {p.get('brandName', 'N/A')}")

        return products

    except Exception as e:
        console.print(f"[bold red]Erro ao buscar produtos: {str(e)}[/bold red]")
        raise


def main():
    console.print("\n[bold purple]=" * 50)
    console.print("[bold purple]Sistema de Similaridade de Produtos")
    console.print("[bold purple]=" * 50)

    try:
        products = fetch_products()

        engine = ProductSimilarityEngine()
        engine.build_index(products)

        console.print(
            "\n[bold cyan]Testando similaridade com produtos de exemplo...[/bold cyan]"
        )
        test_indices = [0, 10, 20]  # Testar com alguns índices de exemplo

        for idx in test_indices:
            console.print("\n[bold purple]=" * 10)
            similar_products = engine.find_similar_products(idx)

            console.print("\n[bold green]Produtos similares encontrados:[/bold green]")
            for item in similar_products:
                similar = item["product"]
                score = item["similarity_score"]
                distance = item["distance"]
                console.print(f"\n[cyan]Produto Similar:[/cyan]")
                console.print(f"Nome: {similar['name']}")
                console.print(f"EAN: {similar['ean']}")
                console.print(f"Categoria: {similar.get('categoryName', 'N/A')}")
                console.print(f"Marca: {similar.get('brandName', 'N/A')}")
                console.print(f"Score de Similaridade: {score:.3f}")
                console.print(f"Distância L2: {distance:.3f}")

        engine.save_embeddings(save_format="all")

        engine.save_embeddings(save_format="numpy")
        engine.save_embeddings(save_format="faiss")

        # load the embeddings
        engine.load_embeddings(load_format="numpy")
        console.print(
            "\n[bold green]Produtos similares encontrados usando embeddings do numpy:[/bold green]"
        )
        for idx in test_indices:
            similar_products = engine.find_similar_products(idx)
            for item in similar_products:
                similar = item["product"]
                score = item["similarity_score"]
                distance = item["distance"]
                console.print(f"\n[cyan]Produto Similar:[/cyan]")
                console.print(f"Nome: {similar['name']}")
                console.print(f"EAN: {similar['ean']}")
                console.print(f"Score de Similaridade: {score:.3f}")
                console.print(f"Distance: {distance:.3f}")
    except Exception as e:
        console.print(f"\n[bold red]Erro durante a execução: {str(e)}[/bold red]")
        raise


if __name__ == "__main__":
    main()
