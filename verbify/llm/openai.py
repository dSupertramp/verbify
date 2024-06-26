import os
from ..config import config
from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from .base_llm import BaseLLM
from typing import Optional, Any
from ..utils.generate_documents import generate_documents


class OpenAILLM(BaseLLM):
    def __init__(self, text: str, api_key: str, model_name: str = "gpt-4o"):
        self.text = text
        self.model_name = model_name
        if not api_key:
            raise ValueError("API key is required for OpenAI.")
        self.api_key = api_key

    def _build_index(self, output_parser=None):
        llm = OpenAI(
            model=self.model_name, api_key=self.api_key, output_parser=output_parser
        )
        embed_model = HuggingFaceEmbedding(
            model_name=config.DEFAULT_EMBEDDINGS,
            cache_folder=f"./rag/embeddings/{config.DEFAULT_EMBEDDINGS.replace('/','_')}",
        )
        documents = generate_documents(content=self.text)
        Settings.llm = llm
        Settings.embed_model = embed_model
        Settings.chunk_size = config.CHUNK_SIZE
        Settings.chunk_overlap = config.CHUNK_OVERLAP
        Settings.context_window = config.CONTEXT_WINDOW
        Settings.num_output = config.NUM_OUTPUT
        if os.path.exists("./rag/vectorstore/openai"):
            vector_store = LanceDBVectorStore(uri="./rag/vectorstore/openai")
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store, persist_dir="./rag/vectorstore/openai"
            )
            index = load_index_from_storage(storage_context=storage_context)
            parser = SimpleNodeParser()
            new_nodes = parser.get_nodes_from_documents(documents)
            index.insert_nodes(new_nodes)
            index = load_index_from_storage(storage_context=storage_context)
        else:
            vector_store = LanceDBVectorStore(uri="./rag/vectorstore/openai")
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex(nodes=documents, storage_context=storage_context)
            index.storage_context.persist(persist_dir="./rag/vectorstore/openai")
        self.index = index

    def query(self, query: str, output_parser: Optional[Any] = None) -> str:
        self._build_index(output_parser=output_parser)
        query_engine = self.index.as_query_engine()
        response = query_engine.query(query)
        return str(response).strip()
