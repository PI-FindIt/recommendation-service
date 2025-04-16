# src/models/recommendation_engine.py
from typing import List, Dict, Any, Optional
import numpy as np
import faiss
from pathlib import Path
import json
import pickle
from rich import print
from rich.console import Console
from rich.table import Table
from sentence_transformers import SentenceTransformer
import torch
from redis import Redis

console = Console()

class RecommendationEngine:
    def __init__(self, load_from: str = 'numpy', timestamp: str = None):
        """
        Inicializa o motor de recomendações.
        
        Args:
            load_from: 'numpy', 'faiss' ou 'redis'
            timestamp: Timestamp específico para carregar. Se None, usa o mais recente
        """
        # Configurar diretórios
        self.storage_dir = Path("storage")
        self.embeddings_dir = self.storage_dir / "embeddings"
        self.index_dir = self.storage_dir / "index"
        self.metadata_dir = self.storage_dir / "metadata"
        
        # Configurar device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        # Carregar modelo
        console.print(f"[yellow]Carregando modelo sentence-transformers...[/yellow]")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.model.to(self.device)
        
        # Carregar dados salvos
        self.load_saved_data(load_from, timestamp)
        
    def load_saved_data(self, load_from: str = 'numpy', timestamp: str|None = None):
        """
        Carrega os dados salvos (embeddings, índice e metadados).
        """
        console.print(f"\n[bold cyan]Carregando dados salvos (formato: {load_from})...[/bold cyan]")
        
        if timestamp is None:
            # Encontrar o arquivo mais recente
            if load_from == 'numpy':
                pattern = "embeddings_*.npy"
                files = list(self.embeddings_dir.glob(pattern))
            elif load_from == 'faiss':
                pattern = "faiss_index_*.idx"
                files = list(self.index_dir.glob(pattern))
            
            if not files:
                raise FileNotFoundError(f"Nenhum arquivo encontrado com padrão {pattern}")
                
            latest_file = max(files, key=lambda x: x.stat().st_mtime)
            timestamp = latest_file.stem.split('_')[1]
        
        # Carregar metadados
        try:
            with open(self.metadata_dir / f"metadata_{timestamp}.json", 'r') as f:
                self.metadata = json.load(f)
                console.print("\n[bold cyan]Metadados carregados:[/bold cyan]")
                for key, value in self.metadata.items():
                    console.print(f"[blue]{key}:[/blue] {value}")
        except FileNotFoundError:
            console.print("[yellow]Arquivo de metadados não encontrado[/yellow]")
            
        # Carregar mapeamento produto-índice
        with open(self.metadata_dir / f"product_mapping_{timestamp}.pkl", 'rb') as f:
            self.product_mapping = pickle.load(f)
            
        if load_from == 'numpy':
            # Carregar embeddings e criar índice FAISS
            embeddings = np.load(self.embeddings_dir / f"embeddings_{timestamp}.npy")
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings.astype('float32'))
            
        elif load_from == 'faiss':
            # Carregar índice FAISS diretamente
            self.index = faiss.read_index(str(self.index_dir / f"faiss_index_{timestamp}.idx"))
            
        elif load_from == 'redis':
            # Conectar ao Redis
            self.redis_client = Redis.from_url("redis://localhost:6379")
            self.redis_index_name = "product_idx"
            
        console.print("[green]✓ Dados carregados com sucesso![/green]")
        
    def get_product_recommendations(
        self,
        product_ean: str,
        k: int = 5,
        filters: Optional[Dict] = None  
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
        for idx, product in self.product_mapping.items():
            if product['ean'] == product_ean:
                product_idx = idx
                break
                
        if product_idx is None:
            raise ValueError(f"Produto com EAN {product_ean} não encontrado!")
            
        # Obter o produto de referência
        product = self.product_mapping[product_idx]
        console.print(f"\n[bold cyan]Buscando recomendações para:[/bold cyan]")
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
                similar_product = self.product_mapping[idx]
                
                # Aplicar filtros se existirem
                if filters:
                    if 'category' in filters and similar_product.get('categoryName') != filters['category']:
                        continue
                    if 'brand' in filters and similar_product.get('brandName') != filters['brand']:
                        continue
                
                recommendations.append({
                    "product": similar_product,
                    "similarity_score": float(1.0 / (1.0 + dist)),
                    "distance": float(dist)
                })
                
        return recommendations[:k]
    
    def get_recommendations_by_text(
        self,
        text_query: str,
        k: int = 5,
        filters: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Obtém recomendações baseadas em uma query de texto.
        
        Args:
            text_query: Texto para buscar produtos similares
            k: Número de recomendações
            filters: Filtros opcionais
        """
        console.print(f"\n[bold cyan]Buscando produtos similares a: {text_query}[/bold cyan]")
        
        # Gerar embedding para a query
        query_embedding = self.model.encode(text_query, device=self.device)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Buscar produtos similares
        distances, indices = self.index.search(query_embedding, k)
        
        # Formatar resultados
        recommendations = []
        for dist, idx in zip(distances[0], indices[0]):
            product = self.product_mapping[idx]
            
            # Aplicar filtros
            if filters:
                if 'category' in filters and product.get('categoryName') != filters['category']:
                    continue
                if 'brand' in filters and product.get('brandName') != filters['brand']:
                    continue
            
            recommendations.append({
                "product": product,
                "similarity_score": float(1.0 / (1.0 + dist)),
                "distance": float(dist)
            })
            
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
                product['name'],
                product['ean'],
                product.get('categoryName', 'N/A'),
                product.get('brandName', 'N/A'),
                f"{score:.3f}"
            )
            
        console.print(table)

def main():
    # Exemplo de uso
    try:
        # Inicializar o engine
        engine = RecommendationEngine(load_from='numpy')
        
        # Exemplo 1: Recomendações por produto
        console.print("\n[bold purple]Exemplo 1: Recomendações por produto[/bold purple]")
        # Usar o EAN de um produto que você sabe que existe
        product_ean = list(engine.product_mapping.values())[0]['ean']  # primeiro produto como exemplo
        recommendations = engine.get_product_recommendations(
            product_ean,
            k=5,
            filters={'category': 'Bebidas'}  # exemplo de filtro
        )
        engine.print_recommendations(recommendations)
        
        # Exemplo 2: Recomendações por texto
        console.print("\n[bold purple]Exemplo 2: Recomendações por texto[/bold purple]")
        text_recommendations = engine.get_recommendations_by_text(
            "café torrado",
            k=5
        )
        engine.print_recommendations(text_recommendations)
        
    except Exception as e:
        console.print(f"[bold red]Erro: {str(e)}[/bold red]")

if __name__ == "__main__":
    main()