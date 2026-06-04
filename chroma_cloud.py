import os
import re
import logging
from typing import List, Dict, Any

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except Exception:
    chromadb = None
    Settings = None
    CHROMADB_AVAILABLE = False

if CHROMADB_AVAILABLE:
    try:
        from chromadb.utils.embedding_functions.chroma_cloud_qwen_embedding_function import (
            ChromaCloudQwenEmbeddingFunction,
            ChromaCloudQwenEmbeddingModel,
        )
    except Exception:
        ChromaCloudQwenEmbeddingFunction = None
        ChromaCloudQwenEmbeddingModel = None

    try:
        from chromadb.utils.embedding_functions.chroma_cloud_splade_embedding_function import (
            ChromaCloudSpladeEmbeddingFunction,
            ChromaCloudSpladeEmbeddingModel,
        )
    except Exception:
        ChromaCloudSpladeEmbeddingFunction = None
        ChromaCloudSpladeEmbeddingModel = None

    try:
        from chromadb.api.types import Schema, VectorIndexConfig, SparseVectorIndexConfig
    except Exception:
        Schema = None
        VectorIndexConfig = None
        SparseVectorIndexConfig = None

    try:
        from chromadb.execution.expression import Knn, Search
    except Exception:
        Knn = None
        Search = None
else:
    ChromaCloudQwenEmbeddingFunction = None
    ChromaCloudQwenEmbeddingModel = None
    ChromaCloudSpladeEmbeddingFunction = None
    ChromaCloudSpladeEmbeddingModel = None
    Schema = None
    VectorIndexConfig = None
    SparseVectorIndexConfig = None
    Knn = None
    Search = None

try:
    from langchain_core.documents import Document
except Exception:
    # Minimal fallback Document shim
    from dataclasses import dataclass

    @dataclass
    class Document:
        page_content: str
        metadata: dict

logger = logging.getLogger(__name__)


def get_client():
    """Return a chroma client configured for Chroma Cloud.
    If chromadb is not installed, raise informative error.
    """
    api_key = os.getenv("CHROMA_API_KEY")
    tenant = os.getenv("CHROMA_TENANT")
    database = os.getenv("CHROMA_DATABASE")
    host = os.getenv("CHROMA_HOST", "api.trychroma.com")
    port = int(os.getenv("CHROMA_PORT", 443))
    ssl = os.getenv("CHROMA_SSL", "true").lower() in ("true", "1", "yes")

    if not CHROMADB_AVAILABLE:
        raise RuntimeError("chromadb SDK not installed. Please install 'chromadb' package or provide a fallback implementation.")

    if not api_key:
        raise RuntimeError("CHROMA_API_KEY is required for Chroma Cloud. Set it in your environment or .env file.")

    try:
        client = chromadb.CloudClient(
            tenant=tenant,
            database=database,
            api_key=api_key,
            cloud_host=host,
            cloud_port=port,
            enable_ssl=ssl,
        )
        return client
    except Exception as e:
        logger.exception("Failed to create chromadb client: %s", e)
        raise


def get_collection_name(base_name: str, dense_embedding_fn=None):
    """Create a stable collection name based on the embedding model."""
    suffix = ""
    model_name = None
    if dense_embedding_fn is not None:
        if hasattr(dense_embedding_fn, "model"):
            model_name = getattr(dense_embedding_fn.model, "name", None)
        if model_name is None and hasattr(dense_embedding_fn, "model_name"):
            model_name = getattr(dense_embedding_fn, "model_name")
        if model_name is None:
            model_name = getattr(dense_embedding_fn.__class__, "__name__", None)
        if model_name:
            model_name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(model_name))
            model_name = model_name.strip("._-")
            if model_name:
                suffix = f"_{model_name}"
    return f"{base_name}{suffix}"


def collection_exists(client, name: str) -> bool:
    try:
        return any(collection.name == name for collection in client.list_collections())
    except Exception:
        return False


def collection_document_count(client, name: str) -> int:
    try:
        coll = client.get_collection(name=name)
        return coll.count()
    except Exception:
        return 0



try:
    import numpy as np
except ImportError:
    np = None


class LangChainEmbeddingWrapper:
    def __init__(self, embeddings, model_name: str = "gemini_embedding_001"):
        self.embeddings = embeddings
        self._model_name = model_name

    def __call__(self, input: List[str]) -> List[Any]:
        embeddings = self.embeddings.embed_documents(input)
        if np is not None:
            return [np.array(emb, dtype=np.float32) for emb in embeddings]
        return embeddings

    def embed_query(self, input: Any) -> List[Any]:
        if isinstance(input, list):
            embeddings = self.embeddings.embed_documents(input)
        else:
            embeddings = [self.embeddings.embed_query(input)]
        if np is not None:
            return [np.array(emb, dtype=np.float32) for emb in embeddings]
        return embeddings

    def name(self) -> str:
        return "langchain-google-embeddings"

    @property
    def model_name(self) -> str:
        return self._model_name


def get_dense_embedding_function(task: str = None):
    # Try Google Generative AI embeddings first (using langchain-google-genai)
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("CHROMA_GOOGLE_GENAI_API_KEY")
        if api_key:
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=api_key
            )
            return LangChainEmbeddingWrapper(embeddings, model_name="gemini_embedding_001")
    except Exception as google_err:
        logger.warning("Failed to initialize Google Generative AI embeddings via langchain: %s", google_err)

    if ChromaCloudQwenEmbeddingFunction is not None and ChromaCloudQwenEmbeddingModel is not None:
        return ChromaCloudQwenEmbeddingFunction(
            model=ChromaCloudQwenEmbeddingModel.QWEN3_EMBEDDING_0p6B,
            task=task,
        )

    # Fallback to Google Generative AI embeddings when available.
    try:
        from chromadb.utils.embedding_functions.google_embedding_function import GoogleGenerativeAiEmbeddingFunction
        api_key = os.getenv("CHROMA_GOOGLE_GENAI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if api_key:
            return GoogleGenerativeAiEmbeddingFunction(api_key=api_key, model_name="models/gemini-embedding-2")
    except Exception:
        pass

    # Fallback to OpenAI embeddings when a key is available.
    try:
        from chromadb.utils.embedding_functions.openai_embedding_function import OpenAIEmbeddingFunction
        api_key = os.getenv("CHROMA_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if api_key:
            return OpenAIEmbeddingFunction(api_key=api_key)
    except Exception:
        pass

    raise RuntimeError(
        "No supported dense embedding function is available. "
        "Install the Chroma Cloud QWEN embedding package or configure a supported embedding service with Google or OpenAI credentials."
    )


def get_sparse_embedding_function(include_tokens: bool = False):
    if ChromaCloudSpladeEmbeddingFunction is None:
        return None
    return ChromaCloudSpladeEmbeddingFunction(
        model=ChromaCloudSpladeEmbeddingModel.SPLADE_PP_EN_V1,
        include_tokens=include_tokens,
    )


def validate_collection_embedding(collection, embedding_fn):
    try:
        if hasattr(embedding_fn, "embed_query"):
            qvecs = embedding_fn.embed_query(["__chroma_dim_probe__"])
        else:
            qvecs = embedding_fn.embed_documents(["__chroma_dim_probe__"])
        qvec = qvecs[0]

        collection.query(
            query_embeddings=[qvec],
            n_results=1,
            include=["documents", "metadatas", "distances"],
        )
        return True
    except chromadb.errors.InvalidArgumentError as e:
        if "dimension" in str(e).lower():
            return False
        raise
    except Exception:
        return True


def get_or_create_collection(
    client,
    name: str,
    metadata: Dict[str, Any] = None,
    dense_embedding_fn=None,
    sparse_embedding_fn=None,
):
    """Get or create a collection on Chroma Cloud using the chromadb client."""
    try:
        kwargs = {"name": name}
        if metadata:
            kwargs["metadata"] = metadata

        if Schema is not None and dense_embedding_fn is not None and sparse_embedding_fn is not None:
            try:
                schema = Schema()
                schema.create_index(
                    config=VectorIndexConfig(
                        embedding_function=dense_embedding_fn,
                    )
                )
                schema.create_index(
                    key="sparse_content",
                    config=SparseVectorIndexConfig(
                        source_key="#document",
                        embedding_function=sparse_embedding_fn,
                    )
                )
                kwargs["schema"] = schema
            except Exception as e:
                logger.warning("Failed to construct Schema: %s. Falling back to default embedding function.", e)
                kwargs["embedding_function"] = dense_embedding_fn
        elif dense_embedding_fn is not None:
            kwargs["embedding_function"] = dense_embedding_fn

        # Recreate the collection if it doesn't match the required schema
        if collection_exists(client, name):
            should_recreate = False
            try:
                existing_coll = client.get_collection(name=name)
                # If we expect a hybrid schema, check if the collection actually has it
                if sparse_embedding_fn is not None:
                    if not hasattr(existing_coll, "schema") or existing_coll.schema is None or "sparse_content" not in existing_coll.schema.keys:
                        logger.warning("Existing collection %s is missing hybrid schema. Deleting it to force recreation.", name)
                        should_recreate = True
                
                # Recreate empty collections to ensure schema aligns
                if not should_recreate:
                    existing_count = existing_coll.count()
                    if existing_count == 0:
                        logger.warning("Existing collection %s is empty. Recreating it to align schema.", name)
                        should_recreate = True
            except Exception as schema_check_error:
                logger.warning("Failed schema check on existing collection %s: %s", name, schema_check_error)
            
            if should_recreate:
                try:
                    client.delete_collection(name=name)
                except Exception as delete_error:
                    logger.warning("Failed to delete collection %s: %s", name, delete_error)

        try:
            coll = client.get_or_create_collection(**kwargs)
        except Exception as first_error:
            logger.warning("get_or_create_collection failed, falling back to create_collection: %s", first_error)
            coll = client.create_collection(**kwargs)

        if dense_embedding_fn is not None and not validate_collection_embedding(coll, dense_embedding_fn):
            logger.warning("Collection %s dimension mismatch; deleting and recreating.", name)
            try:
                client.delete_collection(name=name)
            except Exception as delete_error:
                logger.warning("Failed to delete mismatched collection %s: %s", name, delete_error)
            coll = client.create_collection(**kwargs)

        return coll
    except Exception as e:
        logger.exception("Error accessing/creating collection %s: %s", name, e)
        raise


def upsert_documents(collection, ids: List[str], documents: List[str], metadatas: List[Dict[str, Any]], embeddings: List[List[float]] = None):
    """Upsert documents (and optional embeddings) into a collection.
    Uses the chromadb collection.add API when available.
    """
    try:
        kwargs = {
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
        }
        if embeddings is not None:
            kwargs["embeddings"] = embeddings

        # new chromadb SDK expects collection.add
        return collection.add(**kwargs)
    except Exception as e:
        logger.exception("Failed to upsert documents: %s", e)
        raise


class ChromaCloudRetriever:
    def __init__(self, collection, embedding_fn, k: int = 3, dedupe_by_source: bool = True):
        self.collection = collection
        self.embedding_fn = embedding_fn
        self.k = k
        self.dedupe_by_source = dedupe_by_source

    def _query_collection(self, query: str, qvec):
        return self.collection.query(
            query_embeddings=[qvec],
            query_texts=[query],
            n_results=self.k,
            include=["documents", "metadatas", "distances"],
        )

    def _query_collection_text_only(self, query: str):
        return self.collection.query(
            query_texts=[query],
            n_results=self.k,
            include=["documents", "metadatas", "distances"],
        )

    def get_relevant_documents(self, query: str):
        try:
            # Check if collection supports hybrid search via schema
            has_sparse = False
            if hasattr(self.collection, "schema") and self.collection.schema is not None:
                if hasattr(self.collection.schema, "keys") and "sparse_content" in self.collection.schema.keys:
                    has_sparse = True

            if has_sparse and Search is not None and Knn is not None:
                try:
                    qvec = None
                    if hasattr(self.embedding_fn, "embed_query"):
                        qvecs = self.embedding_fn.embed_query([query])
                        qvec = qvecs[0]
                    elif hasattr(self.embedding_fn, "embed_documents"):
                        qvecs = self.embedding_fn.embed_documents([query])
                        qvec = qvecs[0]
                    elif callable(self.embedding_fn):
                        qvecs = self.embedding_fn([query])
                        qvec = qvecs[0]

                    if qvec is not None:
                        # Convert numpy array to list if needed
                        if hasattr(qvec, "tolist"):
                            qvec = qvec.tolist()
                        elif not isinstance(qvec, list):
                            qvec = list(qvec)

                        dense_rank = Knn(query=qvec, limit=self.k)
                        sparse_rank = Knn(key="sparse_content", query=query, limit=self.k)
                        hybrid_rank = dense_rank + sparse_rank
                        search_expr = Search().rank(hybrid_rank).select_all()
                        
                        result = self.collection.search(search_expr)
                        
                        if hasattr(result, "get"):
                            documents = result.get("documents", [[]])[0]
                            metadatas = result.get("metadatas", [[]])[0]
                        else:
                            documents = getattr(result, "documents", [[]])[0]
                            metadatas = getattr(result, "metadatas", [[]])[0]

                        docs = [Document(page_content=doc_text, metadata=dict(meta or {})) for doc_text, meta in zip(documents, metadatas)]

                        if self.dedupe_by_source and len({(doc.metadata.get("source") or doc.metadata.get("_id", "")) for doc in docs if doc.metadata.get("source") or doc.metadata.get("_id")}) > 1:
                            grouped = {}
                            for doc in docs:
                                source_key = doc.metadata.get("source") or doc.metadata.get("_id", "")
                                if source_key not in grouped:
                                    grouped[source_key] = doc
                            docs = list(grouped.values())

                        return docs[: self.k]
                except Exception as search_err:
                    logger.warning("Hybrid search failed: %s. Falling back to dense-only query.", search_err)

            # Fallback to dense-only query
            qvec = None
            if hasattr(self.embedding_fn, "embed_query"):
                qvecs = self.embedding_fn.embed_query([query])
                qvec = qvecs[0]
            elif hasattr(self.embedding_fn, "embed_documents"):
                qvecs = self.embedding_fn.embed_documents([query])
                qvec = qvecs[0]
            elif callable(self.embedding_fn):
                qvecs = self.embedding_fn([query])
                qvec = qvecs[0]
            else:
                raise RuntimeError("Embedding function does not support query or document embedding.")

            result = None
            if qvec is not None:
                result = self._query_collection(query, qvec)

            if result is None:
                result = self._query_collection_text_only(query)

            if hasattr(result, "get"):
                documents = result.get("documents", [[]])[0]
                metadatas = result.get("metadatas", [[]])[0]
            else:
                documents = getattr(result, "documents", [[]])[0]
                metadatas = getattr(result, "metadatas", [[]])[0]

            docs = [Document(page_content=doc_text, metadata=dict(meta or {})) for doc_text, meta in zip(documents, metadatas)]

            if self.dedupe_by_source and len({(doc.metadata.get("source") or doc.metadata.get("_id", "")) for doc in docs if doc.metadata.get("source") or doc.metadata.get("_id")}) > 1:
                grouped = {}
                for doc in docs:
                    source_key = doc.metadata.get("source") or doc.metadata.get("_id", "")
                    if source_key not in grouped:
                        grouped[source_key] = doc
                docs = list(grouped.values())

            return docs[: self.k]
        except Exception as e:
            logger.exception("ChromaCloudRetriever query failed: %s", e)
            raise


def get_retriever(collection_name: str, embedding_fn, k: int = 3, sparse_embedding_fn=None, dedupe_by_source: bool = False):
    client = get_client()
    db = os.getenv("CHROMA_DATABASE") or collection_name
    db = get_collection_name(db, embedding_fn)
    coll = get_or_create_collection(
        client,
        db,
        dense_embedding_fn=embedding_fn,
        sparse_embedding_fn=sparse_embedding_fn,
    )
    return ChromaCloudRetriever(coll, embedding_fn, k=k, dedupe_by_source=dedupe_by_source)
