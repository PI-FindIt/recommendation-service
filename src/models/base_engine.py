import json
import pickle
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np
import torch
from rich.console import Console
from rich.progress import track
from sentence_transformers import SentenceTransformer

console = Console()


class BaseEngine(ABC):
    def __init__(
        self,
        field_weights: dict[str, float],
        model_name: str = "all-mpnet-base-v2",
        engine_name: str = "BaseEngine",
    ) -> None:
        self.engine_name = engine_name
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.device = torch.device("cpu")
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")

        self.model.to(self.device)

        self.field_weights = field_weights

        self.storage_dir = Path("storage")
        self.embeddings_dir = self.storage_dir / "embeddings"
        self.index_dir = self.storage_dir / "index"
        self.metadata_dir = self.storage_dir / "metadata"

        for directory in [
            self.storage_dir,
            self.embeddings_dir,
            self.index_dir,
            self.metadata_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        self.index = None  # FAISS

        self.data_mapping: dict[int, dict] = {}

    @abstractmethod
    def _generate_field_embeddings(self, obj: dict) -> dict[str, np.ndarray]: ...

    def _combine_embeddings(
        self, field_embeddings: dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Combina os embeddings de diferentes campos usando os pesos definidos.
        """
        first_embedding = next(iter(field_embeddings.values()))
        combined = np.zeros_like(first_embedding)

        used_weights_sum = 0
        for field, embedding_ in field_embeddings.items():
            weight = self.field_weights[field]
            combined += embedding_ * weight
            used_weights_sum += weight

        if used_weights_sum > 0:
            combined = combined / used_weights_sum

        return combined / np.linalg.norm(combined)

    def build_index(self, data: list[dict]) -> None:
        if not data:
            raise ValueError("Lista de dados vazia!")

        console.print(
            f"\n[bold cyan]Iniciando construção do índice para {len(data)} produtos...[/bold cyan]"
        )

        try:
            all_embeddings = []
            self.data_mapping = {}

            for idx, obj in enumerate(track(data, description="Gerando embeddings...")):
                try:
                    field_embeddings = self._generate_field_embeddings(obj)
                    combined = self._combine_embeddings(field_embeddings)

                    if np.isnan(combined).any():
                        raise ValueError("embedding com NaN")
                    if np.isinf(combined).any():
                        raise ValueError("embedding com inf")

                    all_embeddings.append(combined)
                    self.data_mapping[idx] = obj

                except Exception as e:
                    console.print(
                        f"[bold yellow]Erro ao gerar embedding para o produto {idx}: {str(e)}[/bold yellow]"
                    )
                    continue

            if not all_embeddings:
                raise ValueError("Nenhum embedding gerado!")

            embeddings_array = np.array(all_embeddings).astype("float32")
            console.print(
                f"[blue]Shape do array de embeddings: {embeddings_array.shape}[/blue]"
            )
            dimension = embeddings_array.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings_array)

            console.print("[green]✓ Índice construído com sucesso![/green]")
            console.print(f"[blue]Dimensão dos vetores: {dimension}[/blue]")
            console.print(
                f"[blue]Número de produtos indexados: {self.index.ntotal}[/blue]"
            )

        except Exception as e:
            console.print(f"[bold red]Erro ao construir índice: {str(e)}[/bold red]")
            raise

    def save_embeddings(self, save_format: str = "all") -> None:
        if not hasattr(self, "index"):
            raise ValueError(
                "FAISS nao foi inicializado! EXECUTA build_index() primeiro."
            )

        if not hasattr(self, "data_mapping"):
            raise ValueError("data_mapping nao foi inicializado!")

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            console.print(
                f"\n[bold cyan]Salvando embeddings (formato: {save_format})...[/bold cyan]"
            )

            try:
                num_vectors = self.index.ntotal
                dimension = self.index.d

                embeddings = np.empty((num_vectors, dimension), dtype=np.float32)
                for i in range(num_vectors):
                    embeddings[i] = self.index.reconstruct(i)

                if embeddings.size == 0:
                    raise ValueError("Nenhum embedding encontrado no índice FAISS!")

            except Exception as e:
                raise ValueError(
                    f"Erro ao extrair embeddings do índice FAISS: {str(e)}"
                )

            if save_format in ["numpy", "all"]:
                np.save(
                    self.embeddings_dir
                    / f"embeddings_{timestamp}_{self.engine_name}.npy",
                    embeddings,
                )

                with open(
                    self.metadata_dir
                    / f"data_mapping_{timestamp}_{self.engine_name}.pkl",
                    "wb",
                ) as f:
                    pickle.dump(self.data_mapping, f)

                metadata = {
                    "timestamp": timestamp,
                    "model_name": "all-mpnet-base-v2",
                    "embedding_dim": dimension,
                    "num_products": len(self.data_mapping),
                    "field_weights": self.field_weights,
                }

                with open(
                    self.metadata_dir / f"metadata_{timestamp}_{self.engine_name}.json",
                    "w",
                ) as f:
                    json.dump(metadata, f, indent=2)

                console.print("[green]✓ Embeddings salvos em formato numpy[/green]")

            if save_format in ["faiss", "all"]:
                faiss.write_index(
                    self.index,
                    str(
                        self.index_dir
                        / f"faiss_index_{timestamp}_{self.engine_name}.idx"
                    ),
                )
                console.print("[green]✓ Índice FAISS salvo[/green]")

        except Exception as e:
            console.print(f"[bold red]Erro ao salvar embeddings: {str(e)}[/bold red]")
            raise

    def load_embeddings(
        self, timestamp: str | None = None, load_format: str = "numpy"
    ) -> None:

        console.print(
            f"\n[bold cyan]Carregando embeddings (formato: {load_format})...[/bold cyan]"
        )

        if timestamp is None:
            if load_format == "numpy":
                pattern = "embeddings_*.npy"
                files = list(self.embeddings_dir.glob(pattern))
            elif load_format == "faiss":
                pattern = "faiss_index_*.idx"
                files = list(self.index_dir.glob(pattern))
            else:
                raise ValueError(f"Formato não suportado: {load_format}")

            if not files:
                raise FileNotFoundError(
                    f"Nenhum arquivo encontrado com padrão {pattern}"
                )

            latest_file = max(files, key=lambda x: x.stat().st_mtime)
            timestamp = latest_file.stem.split("_")[-4:-2]
            timestamp = "_".join(timestamp).replace(".pkl", "").replace(".json", "")

        try:
            with open(
                self.metadata_dir / f"metadata_{timestamp}_{self.engine_name}.json", "r"
            ) as f:
                self.metadata = json.load(f)
                console.print("\n[bold cyan]Metadados carregados:[/bold cyan]")
                for key, value in self.metadata.items():
                    console.print(f"[blue]{key}:[/blue] {value}")
        except FileNotFoundError:
            console.print("[yellow]Arquivo de metadados não encontrado[/yellow]")

        with open(
            self.metadata_dir / f"data_mapping_{timestamp}_{self.engine_name}.pkl", "rb"
        ) as f:
            self.data_mapping = pickle.load(f)

        if load_format == "numpy":
            embeddings = np.load(
                self.embeddings_dir / f"embeddings_{timestamp}_{self.engine_name}.npy"
            )
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings.astype("float32"))

        elif load_format == "faiss":
            self.index = faiss.read_index(
                str(self.index_dir / f"faiss_index_{timestamp}_{self.engine_name}.idx")
            )
