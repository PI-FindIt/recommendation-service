from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import requests
from rich import print
from rich.console import Console
from rich.progress import track
from rich.table import Table
import time
import torch

console = Console()

class ProductSimilarityEngine:
    def __init__(self):
        console.print("[bold blue]Inicializando ProductSimilarityEngine...[/bold blue]")
        
        # Verificar disponibilidade do MPS
        if torch.backends.mps.is_available():
            console.print("[green]MPS (Metal Performance Shaders) está disponível![/green]")
            self.device = torch.device("mps")
        else:
            console.print("[yellow]MPS não disponível, usando CPU[/yellow]")
            if torch.cuda.is_available():
                console.print("[green]CUDA disponível, usando GPU[/green]")
                self.device = torch.device("cuda")
            else:
                console.print("[yellow]Usando CPU[/yellow]")
                self.device = torch.device("cpu")
                
        console.print(f"[yellow]Carregando modelo sentence-transformers: 'all-MiniLM-L6-v2' no dispositivo: {self.device}[/yellow]")
        
        # Carregar o modelo
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.model.to(self.device)
        
        console.print("[green]✓ Modelo carregado com sucesso![/green]")
        
        # Definir pesos para cada campo
        self.field_weights = {
            'name': 0.35,        # Nome do produto é muito importante
            'generic_name': 0.15, # Nome genérico tem relevância média
            'category': 0.20,    # Categoria é importante para similaridade
            'brand': 0.15,       # Marca tem relevância média
            'keywords': 0.10,    # Keywords ajudam na contextualização
            'ingredients': 0.05   # Ingredientes têm menor peso mas ainda são relevantes
        }
        
        console.print("\n[bold cyan]Pesos definidos para cada campo:[/bold cyan]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Campo")
        table.add_column("Peso")
        for field, weight in self.field_weights.items():
            table.add_row(field, f"{weight:.2f}")
        console.print(table)
        
        self.product_mapping: Dict[int, Dict] = {}
        self.index = None
        
    def _generate_field_embeddings(self, product: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Gera embeddings separados para cada campo do produto.
        """
        embeddings = {}
        
        # Nome do produto
        if product.get('name'):
            embeddings['name'] = self.model.encode(
                product['name'],
                convert_to_numpy=True,
                device=self.device
            )
            
        # Nome genérico
        if product.get('genericName'):
            embeddings['generic_name'] = self.model.encode(
                product['genericName'],
                convert_to_numpy=True,
                device=self.device
            )
            
        # Categoria
        if product.get('categoryName'):
            embeddings['category'] = self.model.encode(
                product['categoryName'],
                convert_to_numpy=True,
                device=self.device
            )
            
        # Marca
        if product.get('brandName'):
            embeddings['brand'] = self.model.encode(
                product['brandName'],
                convert_to_numpy=True,
                device=self.device
            )
            
        # Keywords
        if product.get('keywords') and len(product['keywords']) > 0:
            embeddings['keywords'] = self.model.encode(
                ' '.join(product['keywords']),
                convert_to_numpy=True,
                device=self.device
            )
            
        # Ingredientes
        if product.get('ingredients'):
            embeddings['ingredients'] = self.model.encode(
                product['ingredients'],
                convert_to_numpy=True,
                device=self.device
            )
            
        return embeddings
    
    def _combine_embeddings(self, field_embeddings: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Combina os embeddings de diferentes campos usando os pesos definidos.
        """
        # Inicializar com zeros usando a dimensão do primeiro embedding disponível
        first_embedding = next(iter(field_embeddings.values()))
        combined = np.zeros_like(first_embedding)
        
        # Soma ponderada dos embeddings disponíveis
        used_weights_sum = 0
        for field, embedding in field_embeddings.items():
            weight = self.field_weights[field]
            combined += embedding * weight
            used_weights_sum += weight
            
        # Normalizar pelo soma dos pesos usados
        if used_weights_sum > 0:
            combined = combined / used_weights_sum
            
        # Normalizar o vetor final
        return combined / np.linalg.norm(combined)
    
    def build_index(self, products: List[Dict[str, Any]]):
        """
        Constrói o índice FAISS com os embeddings combinados dos produtos.
        """
        console.print(f"\n[bold cyan]Iniciando construção do índice para {len(products)} produtos...[/bold cyan]")
        
        all_embeddings = []
        
        # Processar cada produto
        for i, product in enumerate(track(products, description="Gerando embeddings")):
            # Gerar embeddings para cada campo
            field_embeddings = self._generate_field_embeddings(product)
            
            # Se é o primeiro produto, mostrar detalhes dos embeddings
            if i == 0:
                console.print("\n[bold green]Detalhes dos embeddings para o primeiro produto:[/bold green]")
                table = Table(show_header=True, header_style="bold magenta")
                table.add_column("Campo")
                table.add_column("Dimensões")
                table.add_column("Norma")
                
                for field, emb in field_embeddings.items():
                    table.add_row(
                        field,
                        str(emb.shape),
                        f"{np.linalg.norm(emb):.3f}"
                    )
                console.print(table)
            
            # Combinar embeddings
            combined = self._combine_embeddings(field_embeddings)
            all_embeddings.append(combined)
        
        # Converter para array numpy
        embeddings_array = np.array(all_embeddings).astype('float32')
        
        # Criar índice FAISS
        console.print("\n[yellow]Construindo índice FAISS...[/yellow]")
        dimension = embeddings_array.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings_array)
        
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
        
        # Gerar embeddings para o produto de consulta
        field_embeddings = self._generate_field_embeddings(product)
        query_embedding = self._combine_embeddings(field_embeddings)
        
        # Mostrar contribuição de cada campo
        console.print("\n[bold green]Contribuição de cada campo para o embedding final:[/bold green]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Campo")
        table.add_column("Peso Efetivo")
        table.add_column("Norma do Embedding")
        
        for field, emb in field_embeddings.items():
            weight = self.field_weights[field]
            norm = np.linalg.norm(emb)
            table.add_row(
                field,
                f"{weight:.3f}",
                f"{norm:.3f}"
            )
        console.print(table)
        
        # Buscar produtos similares
        console.print("\n[yellow]Buscando produtos similares no índice FAISS...[/yellow]")
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k + 1)
        
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
