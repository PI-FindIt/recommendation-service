from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import requests
from rich import print
from rich.console import Console
from rich.progress import track
import time

console = Console()

class ProductSimilarityEngine:
    def __init__(self):
        console.print("[bold blue]Inicializando ProductSimilarityEngine...[/bold blue]")
        console.print(f"[yellow]Carregando modelo sentence-transformers: 'all-mpnet-base-v2'[/yellow]")
        self.model = SentenceTransformer('all-mpnet-base-v2')
        console.print("[green]✓ Modelo carregado com sucesso![/green]")
        
        self.product_mapping: Dict[int, Dict] = {}
        self.index = None
        
    def _create_product_description(self, product: Dict[str, Any]) -> str:
        """
        Cria uma descrição rica do produto para gerar embeddings mais significativos.
        """
        parts = []
        
        # Nome e nome genérico
        parts.append(f"nome: {product['name']}")
        parts.append(f"nome genérico: {product['genericName']}")
        
        # Categoria e marca se disponíveis
        if product.get('categoryName'):
            parts.append(f"categoria: {product['categoryName']}")
        if product.get('brandName'):
            parts.append(f"marca: {product['brandName']}")
            
        # Keywords se disponíveis
        if product.get('keywords'):
            parts.append(f"palavras-chave: {' '.join(product['keywords'])}")
            
        # Ingredientes se disponíveis
        if product.get('ingredients'):
            parts.append(f"ingredientes: {product['ingredients']}")
            
        final_description = " | ".join(filter(None, parts))
        return final_description
    
    def build_index(self, products: List[Dict[str, Any]]):
        """
        Constrói o índice FAISS com os embeddings dos produtos.
        """
        console.print(f"\n[bold cyan]Iniciando construção do índice para {len(products)} produtos...[/bold cyan]")
        
        # Criar descrições ricas para cada produto
        console.print("\n[yellow]Gerando descrições ricas para cada produto...[/yellow]")
        descriptions = []
        for product in track(products, description="Processando produtos"):
            desc = self._create_product_description(product)
            descriptions.append(desc)
            
        # Mostrar algumas descrições de exemplo
        console.print("\n[bold green]Exemplos de descrições geradas:[/bold green]")
        for i in range(min(3, len(descriptions))):
            console.print(f"[cyan]Produto {i+1}:[/cyan] {descriptions[i][:200]}...")
        
        # Gerar embeddings
        console.print("\n[yellow]Gerando embeddings para as descrições...[/yellow]")
        start_time = time.time()
        embeddings = self.model.encode(descriptions, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        end_time = time.time()
        
        console.print(f"[green]✓ Embeddings gerados em {end_time - start_time:.2f} segundos[/green]")
        console.print(f"[blue]Forma do tensor de embeddings: {embeddings.shape}[/blue]")
        
        # Criar índice FAISS
        console.print("\n[yellow]Construindo índice FAISS...[/yellow]")
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        
        # Mapear índices para produtos
        self.product_mapping = {i: product for i, product in enumerate(products)}
        
        console.print(f"[bold green]✓ Índice construído com sucesso![/bold green]")
        console.print(f"[blue]Dimensão dos vetores: {dimension}[/blue]")
        console.print(f"[blue]Número total de produtos indexados: {self.index.ntotal}[/blue]")
        
    def find_similar_products(self, product_idx: int, k: int = 5) -> List[Dict[str, Any]]:
        """
        Encontra os k produtos mais similares ao produto dado.
        """
        if self.index is None:
            raise ValueError("Índice não foi construído ainda!")
            
        console.print(f"\n[bold cyan]Buscando {k} produtos similares...[/bold cyan]")
        
        # Obter o produto de referência
        product = self.product_mapping[product_idx]
        console.print(f"\n[yellow]Produto de referência:[/yellow]")
        console.print(f"Nome: {product['name']}")
        console.print(f"EAN: {product['ean']}")
        console.print(f"Categoria: {product.get('categoryName', 'N/A')}")
        console.print(f"Marca: {product.get('brandName', 'N/A')}")
        
        # Gerar embedding para a consulta
        query_desc = self._create_product_description(product)
        console.print("\n[yellow]Gerando embedding para consulta...[/yellow]")
        query_embedding = self.model.encode([query_desc])[0].reshape(1, -1).astype('float32')
        
        # Buscar produtos similares
        console.print("\n[yellow]Buscando produtos similares no índice FAISS...[/yellow]")
        distances, indices = self.index.search(query_embedding, k + 1)
        
        # Formatar resultados
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != product_idx:  # Não incluir o próprio produto
                similar_product = self.product_mapping[idx]
                similarity_score = float(1.0 / (1.0 + dist))
                results.append({
                    "product": similar_product,
                    "similarity_score": similarity_score,
                    "distance": float(dist)
                })
        
        return results

def fetch_products(limit: int = 14000) -> List[Dict[str, Any]]:
    """
    Busca produtos da API GraphQL.
    """
    console.print("\n[bold cyan]Iniciando busca de produtos...[/bold cyan]")
    
    query = """
    query Products($filters: ProductFilter!) {
        products(filters: $filters) {
            ean
            name
            genericName
            brandName
            categoryName
            keywords
            ingredients
            nutriScore
        }
    }
    """
    
    variables = {
        "filters": {}
    }
    
    console.print(f"[yellow]Fazendo requisição GraphQL para buscar até {limit} produtos...[/yellow]")
    
    try:
        response = requests.post(
            "http://localhost",
            json={"query": query, "variables": variables}
        )
        
        if response.status_code != 200:
            console.print(f"[bold red]Erro na requisição: Status {response.status_code}[/bold red]")
            raise Exception(f"Falha ao buscar produtos: {response.text}")
            
        data = response.json()
        products = data["data"]["products"][:limit]
        
        console.print(f"[green]✓ Busca concluída com sucesso![/green]")
        console.print(f"[blue]Total de produtos encontrados: {len(products)}[/blue]")
        
        # Mostrar alguns exemplos
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
        # Buscar produtos
        products = fetch_products()
        
        # Criar e treinar o engine
        engine = ProductSimilarityEngine()
        engine.build_index(products)
        
        # Testar com alguns produtos
        console.print("\n[bold cyan]Testando similaridade com produtos de exemplo...[/bold cyan]")
        test_indices = [0, 10, 20]  # Testar com alguns índices de exemplo
        
        for idx in test_indices:
            console.print("\n[bold purple]=" * 50)
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
    
    except Exception as e:
        console.print(f"\n[bold red]Erro durante a execução: {str(e)}[/bold red]")
        raise

if __name__ == "__main__":
    main()
