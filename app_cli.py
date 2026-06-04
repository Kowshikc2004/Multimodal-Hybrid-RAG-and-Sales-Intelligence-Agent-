import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import chroma_cloud
import rag_service

load_dotenv(override=True)

DEFAULT_MODEL = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")


def run_cli():
    llm = rag_service.get_llm()
    dense_fn = chroma_cloud.get_dense_embedding_function(task=None)
    sparse_fn = chroma_cloud.get_sparse_embedding_function()
    base_name = os.getenv("CHROMA_DATABASE") or "Jokaan_RAG_exe"
    collection_name = chroma_cloud.get_collection_name(base_name, dense_fn)
    client = chroma_cloud.get_client()

    if not chroma_cloud.collection_exists(client, collection_name) or chroma_cloud.collection_document_count(client, collection_name) == 0:
        print("Building vector database from files and web... this may take a minute.")
        rag_service.build_vector_database()

    retriever = chroma_cloud.get_retriever(
        base_name,
        embedding_fn=dense_fn,
        k=3,
        sparse_embedding_fn=sparse_fn,
        dedupe_by_source=True,
    )

    print("Knowledge base CLI is ready. Type a question or 'exit' to quit.")
    while True:
        query = input("\n> ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            break

        docs = retriever.get_relevant_documents(query)
        provider = os.getenv("LLM_PROVIDER", "gemini").lower()
        if provider == "ollama":
            docs = docs[:2]
        if docs:
            prompt_text = rag_service.build_prompt(query, docs)
        else:
            prompt_text = rag_service.build_no_context_prompt(query)

        llm_instance = rag_service.get_llm_for_docs(docs)
        try:
            try:
                answer = llm_instance.invoke(prompt_text)
            except AttributeError:
                answer = llm_instance(prompt_text)
        except Exception as e:
            provider = os.getenv("LLM_PROVIDER", "gemini").lower()
            if provider == "gemini":
                print(f"Gemini error: {e}. Falling back to local Ollama model...")
                local_llm = rag_service.OllamaChatModel(
                    model_name=os.getenv("OLLAMA_MODEL", "llama3.2"),
                    temperature=0.1
                )
                answer = local_llm.invoke(prompt_text)
            else:
                raise e

        answer_text = rag_service.extract_response_text(answer)
        print("\n=== Answer ===\n")
        print(answer_text)


if __name__ == "__main__":
    run_cli()
