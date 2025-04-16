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
import os
import pickle
from datetime import datetime
import json
from pathlib import Path
from redis import Redis
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

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
                
        console.print(f"[yellow]Carregando modelo sentence-transformers: 'all-mpnet-base-v2' no dispositivo: {self.device}[/yellow]")
        
        # Carregar o modelo
        self.model = SentenceTransformer('all-mpnet-base-v2')
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
        
        # Configurar diretórios para armazenamento
        self.storage_dir = Path("storage")
        self.embeddings_dir = self.storage_dir / "embeddings"
        self.index_dir = self.storage_dir / "index"
        self.metadata_dir = self.storage_dir / "metadata"
        
        # Criar diretórios se não existirem
        for directory in [self.storage_dir, self.embeddings_dir, self.index_dir, self.metadata_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
        # Configurar Redis (opcional, será inicializado apenas se necessário)
        self.redis_client = None
        self.redis_index_name = "product_idx"
        
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
        Constrói o índice FAISS com os embeddings dos produtos.
        """
        if not products:
            raise ValueError("Lista de produtos está vazia!")
        
        console.print(f"\n[bold cyan]Iniciando construção do índice para {len(products)} produtos...[/bold cyan]")
        
        try:
            # Gerar embeddings para cada campo
            all_embeddings = []
            self.product_mapping = {}
            
            for i, product in enumerate(track(products, description="Gerando embeddings")):
                try:
                    # Gerar embeddings para os campos do produto
                    field_embeddings = self._generate_field_embeddings(product)
                    
                    # Combinar embeddings
                    combined = self._combine_embeddings(field_embeddings)
                    
                    # Verificar se o embedding é válido
                    if np.isnan(combined).any():
                        raise ValueError("Embedding contém valores NaN")
                    if np.isinf(combined).any():
                        raise ValueError("Embedding contém valores infinitos")
                    
                    # Adicionar ao array de embeddings
                    all_embeddings.append(combined)
                    
                    # Mapear produto
                    self.product_mapping[i] = product
                    
                    # Debug: mostrar progresso a cada 100 produtos
                    if i % 100 == 0:
                        console.print(f"[blue]Processados {i} produtos[/blue]")
                    
                except Exception as e:
                    console.print(f"[yellow]Aviso: Erro ao processar produto {i}: {str(e)}[/yellow]")
                    continue
            
            if not all_embeddings:
                raise ValueError("Nenhum embedding foi gerado!")
            
            # Converter para array numpy
            embeddings_array = np.array(all_embeddings).astype('float32')
            
            console.print(f"[blue]Shape do array de embeddings: {embeddings_array.shape}[/blue]")
            
            # Criar e popular índice FAISS
            dimension = embeddings_array.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings_array)
            
            console.print(f"[green]✓ Índice construído com sucesso![/green]")
            console.print(f"[blue]Dimensão dos vetores: {dimension}[/blue]")
            console.print(f"[blue]Número de produtos indexados: {self.index.ntotal}[/blue]")
            
        except Exception as e:
            console.print(f"[bold red]Erro ao construir índice: {str(e)}[/bold red]")
            raise
    
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

    def _initialize_redis(self, redis_url: str = "redis://localhost:6379"):
        """
        Inicializa a conexão com Redis e cria o índice de vetores
        """
        try:
            self.redis_client = Redis.from_url(redis_url)
            
            # Definir schema do índice
            schema = (
                TextField("$.ean", as_name="ean"),
                TextField("$.name", as_name="name"),
                TextField("$.category", as_name="category"),
                TextField("$.brand", as_name="brand"),
                VectorField("$.embedding", 
                           "FLAT", {
                               "TYPE": "FLOAT32",
                               "DIM": 768,  # dimensão do modelo all-mpnet-base-v2
                               "DISTANCE_METRIC": "L2"
                           }, as_name="embedding")
            )
            
            try:
                # Tentar criar o índice
                self.redis_client.ft(self.redis_index_name).create_index(
                    fields=schema,
                    definition=IndexDefinition(prefix=["prod:"], index_type=IndexType.JSON)
                )
            except Exception as e:
                # Índice pode já existir, isso é ok
                console.print(f"[yellow]Nota: {str(e)}[/yellow]")
                
            console.print("[green]✓ Conexão com Redis estabelecida com sucesso![/green]")
            
        except Exception as e:
            console.print(f"[red]Erro ao conectar com Redis: {e}[/red]")
            self.redis_client = None

    def save_embeddings(self, save_format: str = "all"):
        """
        Salva os embeddings e metadados em diferentes formatos.
        
        Args:
            save_format: Formato de salvamento ('numpy', 'faiss', 'redis', 'all')
        """
        # Verificações iniciais
        if not hasattr(self, 'index') or self.index is None:
            raise ValueError("Índice FAISS não foi inicializado! Execute build_index primeiro.")
        
        if not hasattr(self, 'product_mapping') or not self.product_mapping:
            raise ValueError("Mapeamento de produtos não foi inicializado!")
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            console.print(f"\n[bold cyan]Salvando embeddings (formato: {save_format})...[/bold cyan]")

            # Extrair embeddings do índice FAISS
            try:
                # Obter o número total de vetores e dimensão
                num_vectors = self.index.ntotal
                dimension = self.index.d
                
                # Criar um array para armazenar os vetores
                embeddings = np.empty((num_vectors, dimension), dtype=np.float32)
                
                # Reconstruir os vetores do índice
                for i in range(num_vectors):
                    embeddings[i] = self.index.reconstruct(i)
                    
                if embeddings.size == 0:
                    raise ValueError("Nenhum embedding encontrado no índice FAISS!")
                    
            except Exception as e:
                raise ValueError(f"Erro ao extrair embeddings do índice FAISS: {str(e)}")

            if save_format in ['numpy', 'all']:
                # Salvar embeddings em formato numpy
                np.save(
                    self.embeddings_dir / f"embeddings_{timestamp}.npy",
                    embeddings
                )
                
                # Salvar mapeamento produto-índice
                with open(self.metadata_dir / f"product_mapping_{timestamp}.pkl", 'wb') as f:
                    pickle.dump(self.product_mapping, f)
                    
                # Salvar configurações e metadados
                metadata = {
                    'timestamp': timestamp,
                    'model_name': 'all-mpnet-base-v2',
                    'embedding_dim': dimension,
                    'num_products': len(self.product_mapping),
                    'field_weights': self.field_weights
                }
                
                with open(self.metadata_dir / f"metadata_{timestamp}.json", 'w') as f:
                    json.dump(metadata, f, indent=2)
                    
                console.print("[green]✓ Embeddings salvos em formato numpy[/green]")

            if save_format in ['faiss', 'all']:
                # Salvar índice FAISS
                faiss.write_index(
                    self.index,
                    str(self.index_dir / f"faiss_index_{timestamp}.idx")
                )
                console.print("[green]✓ Índice FAISS salvo[/green]")

            if save_format in ['redis', 'all']:
                # Inicializar Redis se necessário
                if not self.redis_client:
                    self._initialize_redis()
                    
                if self.redis_client:
                    # Salvar embeddings no Redis
                    pipe = self.redis_client.pipeline()
                    
                    for idx, (product_id, product) in enumerate(self.product_mapping.items()):
                        vector = embeddings[idx].tolist()
                        
                        # Criar documento com metadata e embedding
                        data = {
                            'ean': product['ean'],
                            'name': product['name'],
                            'category': product.get('categoryName', ''),
                            'brand': product.get('brandName', ''),
                            'timestamp': timestamp,
                            'embedding': vector
                        }

                        # Adicionar ao Redis
                        pipe.json().set(f"prod:{product['ean']}", '$', data)
                        console.print(f"[green]Produto {product['ean']} salvo no Redis[/green]")
                    
                    # Executar todas as operações
                    pipe.execute()
                    console.print("[green]✓ Embeddings salvos no Redis[/green]")

        except Exception as e:
            console.print(f"[bold red]Erro ao salvar embeddings: {str(e)}[/bold red]")
            raise

    def load_embeddings(self, timestamp: str = None, load_format: str = 'numpy'):
        """
        Carrega embeddings salvos anteriormente.
        
        Args:
            timestamp: Timestamp específico para carregar. Se None, usa o mais recente
            load_format: Formato para carregar ('numpy', 'faiss', 'redis')
        """
        console.print(f"\n[bold cyan]Carregando embeddings (formato: {load_format})...[/bold cyan]")

        if timestamp is None:
            # Encontrar o arquivo mais recente baseado no formato
            if load_format == 'numpy':
                pattern = "embeddings_*.npy"
                files = list(self.embeddings_dir.glob(pattern))
            elif load_format == 'faiss':
                pattern = "faiss_index_*.idx"
                files = list(self.index_dir.glob(pattern))
            else:
                raise ValueError(f"Formato não suportado: {load_format}")

            if not files:
                raise FileNotFoundError(f"Nenhum arquivo encontrado com padrão {pattern}")
                
            latest_file = max(files, key=lambda x: x.stat().st_mtime)
            timestamp = latest_file.stem.split('_')[1]

        if load_format == 'numpy':
            # Carregar embeddings
            embeddings = np.load(self.embeddings_dir / f"embeddings_{timestamp}.npy")
            
            # Carregar mapeamento produto-índice
            with open(self.metadata_dir / f"product_mapping_{timestamp}.pkl", 'rb') as f:
                self.product_mapping = pickle.load(f)
                
            # Criar novo índice FAISS
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings.astype('float32'))
            
            console.print("[green]✓ Embeddings carregados do numpy[/green]")

        elif load_format == 'faiss':
            # Carregar índice FAISS
            self.index = faiss.read_index(
                str(self.index_dir / f"faiss_index_{timestamp}.idx")
            )
            
            # Carregar mapeamento produto-índice
            with open(self.metadata_dir / f"product_mapping_{timestamp}.pkl", 'rb') as f:
                self.product_mapping = pickle.load(f)
                
            console.print("[green]✓ Índice FAISS carregado[/green]")

        elif load_format == 'redis':
            if not self.redis_client:
                self._initialize_redis()
                
            if self.redis_client:
                console.print("[green]✓ Conectado ao índice Redis[/green]")
                # Não precisamos carregar nada explicitamente do Redis
                # Os vetores já estão disponíveis para consulta
            
        # Carregar metadados
        try:
            with open(self.metadata_dir / f"metadata_{timestamp}.json", 'r') as f:
                metadata = json.load(f)
                console.print("\n[bold cyan]Metadados do modelo carregado:[/bold cyan]")
                for key, value in metadata.items():
                    console.print(f"[blue]{key}:[/blue] {value}")
        except FileNotFoundError:
            console.print("[yellow]Arquivo de metadados não encontrado[/yellow]")

    def find_similar_products_redis(self, product_ean: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Encontra produtos similares usando Redis Vector Search
        """
        if not self.redis_client:
            raise ValueError("Redis não está configurado!")
            
        # Obter o embedding do produto de consulta
        product = self.product_mapping[product_ean]
        query_embedding = self.model.encode(product['name']).astype('float32').tolist()
        
        # Construir a query
        query = (
            Query(f'*=>[KNN {k + 1} @embedding $vec AS score]')
            .dialect(2)
            .paging(0, k + 1)
            .return_fields('ean', 'name', 'category', 'brand', 'score')
            .sort_by('score')
        )
        
        # Executar a busca
        results = self.redis_client.ft(self.redis_index_name).search(
            query,
            {'vec': query_embedding}
        )
        
        # Formatar resultados
        similar_products = []
        for doc in results.docs:
            if doc.ean != product_ean:  # Não incluir o próprio produto
                similar_products.append({
                    "product": {
                        "ean": doc.ean,
                        "name": doc.name,
                        "category": doc.category,
                        "brand": doc.brand
                    },
                    "similarity_score": 1 / (1 + float(doc.score))  # Converter distância em similaridade
                })
                
        return similar_products[:k]

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
            "http://100.109.29.71",
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
            
        # Salvar em todos os formatos
        engine.save_embeddings(save_format='all')

        # Ou escolher um formato específico
        engine.save_embeddings(save_format='numpy')
        engine.save_embeddings(save_format='faiss')
        engine.save_embeddings(save_format='redis')
        
        # Encontrar produtos similares usando Redis
        similar_products = engine.find_similar_products_redis('5600954501582')
        console.print("\n[bold green]Produtos similares encontrados usando Redis:[/bold green]")
        for item in similar_products:
            similar = item["product"]
            score = item["similarity_score"]
            console.print(f"\n[cyan]Produto Similar:[/cyan]")
            console.print(f"Nome: {similar['name']}")

    except Exception as e:
        console.print(f"\n[bold red]Erro durante a execução: {str(e)}[/bold red]")
        raise

if __name__ == "__main__":
    main()
    