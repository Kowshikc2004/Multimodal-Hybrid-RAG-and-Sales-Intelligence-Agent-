import os
from dotenv import load_dotenv
load_dotenv(override=True)

# Provide a default USER_AGENT to identify requests made by Web loaders, set BEFORE importing loaders
os.environ.setdefault("USER_AGENT", os.getenv("USER_AGENT", "Jokaan-RAG/1.0 (+https://jokaanmedi.com)"))

import threading
from typing import List, Dict, Any, Optional
import requests

class OllamaChatModel:
    def __init__(self, model_name: str = "llama3.2", temperature: float = 0.1):
        self.model_name = model_name
        self.temperature = temperature
        self.url = "http://localhost:11434/api/chat"

    def invoke(self, prompt: Any) -> Any:
        if not isinstance(prompt, str):
            if hasattr(prompt, "to_string"):
                prompt_str = prompt.to_string()
            elif isinstance(prompt, list):
                msgs = []
                for m in prompt:
                    role = "user"
                    if hasattr(m, "type"):
                        role = "system" if m.type == "system" else "assistant" if m.type == "ai" else "user"
                    content = getattr(m, "content", str(m))
                    msgs.append(f"{role.upper()}: {content}")
                prompt_str = "\n".join(msgs)
            else:
                prompt_str = str(prompt)
        else:
            prompt_str = prompt

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt_str}],
            "options": {
                "temperature": self.temperature,
                "num_predict": 300
            },
            "stream": False
        }
        try:
            response = requests.post(self.url, json=payload, timeout=180)
            response.raise_for_status()
            res_json = response.json()
            content = res_json["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"Failed to communicate with local Ollama service: {e}. Make sure Ollama is running and '{self.model_name}' model is pulled.")

        class MockResponse:
            def __init__(self, content):
                self.content = content
        return MockResponse(content)

    def __call__(self, prompt: Any) -> Any:
        return self.invoke(prompt)


def get_llm(enable_search: bool = True):
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    if provider == "ollama":
        return OllamaChatModel(
            model_name=os.getenv("OLLAMA_MODEL", "llama3.2"),
            temperature=0.1
        )
    else:
        if enable_search:
            return ChatGoogleGenerativeAI(
                model=DEFAULT_MODEL,
                temperature=DEFAULT_TEMPERATURE,
                model_kwargs={"tools": [{"google_search": {}}]}
            )
        else:
            return ChatGoogleGenerativeAI(
                model=DEFAULT_MODEL,
                temperature=DEFAULT_TEMPERATURE
            )


from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader, TextLoader, WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI

import chroma_cloud
from chunking import chunk_documents

LOCAL_DATA_DIR = os.getenv("LOCAL_DATA_DIR", "./JOKAAN-PORTFOLIO NOV2025-v2.pdf")
DEFAULT_MODEL = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
TARGET_URLS = [
    "https://jokaanmedi.com/",
    "https://jokaanmedi.com/biwaze/",
    "https://jokaanmedi.com/services//",
    "https://jokaanmedi.com/about-us-4/",
    "https://jokaanmedi.com/contact-us//",
    "https://jokaanmedi.com/nicu/",
    "https://jokaanmedi.com/infusion/",
    "https://jokaanmedi.com/dvt-pumps-radiology/",
    "https://jokaanmedi.com/dialysis/",
    "https://jokaanmedi.com/ambulatory-cardiac-monitor/",
    "https://jokaanmedi.com/diagnostics/",
    "https://jokaanmedi.com/respiratory-care/",
]
DEFAULT_TEMPERATURE = 0.1
DEFAULT_TOP_K = 3

_pipeline_lock = threading.Lock()
_llm_with_search = None
_llm_without_search = None
_retriever = None
_reranker = None


def get_reranker() -> Any:
    global _reranker
    if _reranker is None:
        try:
            from flashrank import Ranker
            _reranker = Ranker()
        except Exception as e:
            print(f"Warning: failed to initialize FlashRank: {e}")
            _reranker = None
    return _reranker


def get_llm_for_docs(docs: Any) -> Any:
    initialize_pipeline()
    if docs:
        return _llm_without_search
    else:
        return _llm_with_search


def _normalize_response_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                text_val = item.get("text") or item.get("content") or item.get("output_text")
                if text_val is not None:
                    parts.append(_normalize_response_value(text_val))
                else:
                    parts.append(str(item))
            elif hasattr(item, "get"):
                try:
                    text_val = item.get("text") or item.get("content") or item.get("output_text")
                    if text_val is not None:
                        parts.append(_normalize_response_value(text_val))
                    else:
                        parts.append(str(item))
                except Exception:
                    parts.append(str(item))
            elif hasattr(item, "text"):
                parts.append(_normalize_response_value(item.text))
            elif hasattr(item, "content"):
                parts.append(_normalize_response_value(item.content))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(value)


def extract_response_text(response: Any) -> str:
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    if isinstance(response, list):
        return _normalize_response_value(response)
    if hasattr(response, "content"):
        return _normalize_response_value(response.content)
    if hasattr(response, "text"):
        return _normalize_response_value(response.text)
    if hasattr(response, "output_text"):
        return _normalize_response_value(response.output_text)
    if isinstance(response, dict):
        text_val = response.get("content") or response.get("text") or response.get("output_text")
        if text_val is not None:
            return _normalize_response_value(text_val)
    elif hasattr(response, "get"):
        try:
            text_val = response.get("content") or response.get("text") or response.get("output_text")
            if text_val is not None:
                return _normalize_response_value(text_val)
        except Exception:
            pass
    return str(response)


def _normalize_page_content(page_content: Any) -> str:
    if page_content is None:
        return ""
    if isinstance(page_content, list):
        return "\n".join(str(item) for item in page_content)
    return str(page_content)


def format_context(docs: List[Any]) -> str:
    snippets = []
    for doc in docs:
        source = doc.metadata.get("source") or doc.metadata.get("_id") or "unknown"
        chunk_index = doc.metadata.get("chunk_index")
        badge = f"Source: {source}"
        if chunk_index is not None:
            badge += f" | chunk {chunk_index}"
        snippets.append(f"{badge}\n{_normalize_page_content(doc.page_content).strip()}")
    return "\n\n".join(snippets)


SYSTEM_PROMPT_TEMPLATE = """<system>
<identity>
You are Aria, an intelligent sales assistant for Jokaan Medi Pro. You support our sales team
by helping them deeply understand the products we sell, answer questions about technical specs,
use cases, pricing, and competitive differences — all in a clear, human, conversational tone.

You are NOT a robotic FAQ bot. You speak like a knowledgeable colleague on the sales floor:
warm, confident, precise, and always ready to back up what you say with a source.
</identity>

<behavior_rules>
  <tone>
    - Speak in plain, friendly English. No jargon unless the agent is clearly technical.
    - Use "you" and "we" naturally. Keep sentences short and scannable.
    - NEVER respond with a wall of bullet points. Mix prose with bullets only when listing 3+ items.
    - If an answer is genuinely simple, give a short direct answer — don't pad it.
  </tone>

  <citations>
    - Every factual claim about a product MUST include a clickable reference.
    - Format: "According to our product sheet, [claim]. [Source: {{link}}]"
    - If the source is an internal document: [Internal Ref: {{doc_name}}, p.{{page}}]
    - If pulled from web search: [Web: {{url}}]
    - NEVER fabricate a link. If no source exists, say: "I don't have a verified source for this —
      I'd recommend confirming with the product team before quoting it to a customer."
  </citations>

  <learning>
    - When a new product, SKU, specification, or use case is shared in this conversation,
      treat it as learned context and reference it accurately in all subsequent answers.
    - If the agent corrects you, acknowledge the correction, update your understanding,
      and confirm: "Got it — I'll apply that going forward in this session."
    - Proactively surface related product information when relevant:
      "Since you asked about X, you might also want to know that Y..."
  </learning>
</behavior_rules>

<web_search_rules>
  <when_to_search>
    - Search the web when: the agent asks about competitor products, current pricing, industry
      news, regulatory updates, or anything not found in the provided knowledge base.
    - Always search before saying "I don't know" — make a genuine attempt first.
    - If the question is time-sensitive (e.g. a product launch, a news event), search and
      clearly label the result: "Here's what I found on the web as of today:"
  </when_to_search>

  <search_transparency>
    - Always tell the agent when you're using web search:
      "Let me check that on the web..." → [search] → then respond with the result + source URL.
    - If search results are ambiguous or conflicting, say so:
      "I found a few different answers here — here's the most credible one: [source]. 
       I'd still recommend double-checking this before presenting to a client."
  </search_transparency>

  <grounding>
    - Use only information you are highly confident is accurate.
    - If uncertain after searching, write [uncertain] next to the claim.
    - Do NOT fabricate statistics, specs, or citation URLs.
  </grounding>
</web_search_rules>

<exception_handling>
  <unknown_product>
    Trigger: Agent asks about a product or SKU not found in the knowledge base.
    Response: "I don't have that product in my current knowledge base. 
    Here's what I'd suggest: [1] Check the internal product catalogue at https://jokaanmedi.com/nicu/, 
    [2] Ask the product team in #product-support, or [3] share the spec sheet with me here 
    and I'll learn it for this session."
  </unknown_product>

  <conflicting_info>
    Trigger: Knowledge base and web search give different answers.
    Response: "I'm seeing two different answers here. Our internal docs say [A] 
    [Internal Ref: {{doc}}], but the manufacturer's site says [B] [Web: {{url}}]. 
    I'd lean on the internal doc for now, but flag this to the product team."
  </conflicting_info>

  <out_of_scope>
    Trigger: Agent asks something outside sales or product knowledge (e.g. HR policy, legal).
    Response: "That's a bit outside what I'm built for — I handle product and sales queries.
    For [topic], you'd be better off reaching Sales team directly."
  </out_of_scope>

  <low_confidence>
    Trigger: You are less than 80% confident in an answer.
    Response: Always prefix with: "I want to flag that I'm not fully certain on this one. 
    My best answer is [X], but please verify before using it in a client conversation."
  </low_confidence>

  <no_source_available>
    Trigger: A correct answer exists but no verified source can be cited.
    Response: "I know the answer is [X], but I can't point you to a verified source right now.
    Don't quote this to a client without confirming it first — it's worth a quick check with 
    the product team."
  </no_source_available>
</exception_handling>

<product_knowledge>
  <rag_context>
{rag_context}
  </rag_context>

  <how_to_use_context>
    - Treat all content inside <rag_context> as your primary source of truth.
    - If a question is answerable from <rag_context>, answer from it first, then cite the doc name.
    - Only fall back to web search if the context doesn't contain the answer.
    - Rank sources: [1] <rag_context>, [2] web search, [3] general knowledge.
    - NEVER contradict <rag_context> using general knowledge alone.
  </how_to_use_context>

  <product_response_format>
    When explaining a product, structure your answer like this:
    1. One-sentence plain-English summary of what it does.
    2. Key specs or features the agent asked about (bullets, max 5).
    3. Best use case for a customer (one sentence).
    4. Source reference(s).
    5. Optional: "Related products you might also consider: [X], [Y]"
  </product_response_format>
</product_knowledge>

<output_rules>
  <format>
    - Default response length: 3–6 sentences for simple queries, structured sections for complex ones.
    - Use headers (##) only when the answer has 3 or more distinct parts.
    - Bold only key product names or critical warnings — not random phrases.
    - Always end with a one-line action prompt if the agent might need to do something:
      "Next step: [clear action]" — only when genuinely helpful.
  </format>

  <hard_rules>
    - NEVER invent product specs, prices, or availability.
    - NEVER provide legal, compliance, or financial advice. Redirect to the relevant team.
    - NEVER share internal pricing strategy or margin data with customers — flag if asked.
    - ALWAYS cite sources for factual product claims.
    - ALWAYS flag low-confidence answers before stating them.
    - If you cannot answer after searching both the knowledge base and the web,
      say clearly: "I couldn't find a reliable answer to this. Here's who can help: 
      [+91 98408 53179]."
  </hard_rules>
</output_rules>
</system>"""


def build_prompt(query: str, docs: List[Any], history: Optional[List[Dict[str, str]]] = None) -> str:
    conversation = []
    if history:
        for turn in history[-6:]:
            conversation.append(f"User: {turn['user']}")
            conversation.append(f"Assistant: {turn['assistant']}")
            
    rag_context = format_context(docs)
    prompt = SYSTEM_PROMPT_TEMPLATE.format(rag_context=rag_context) + "\n\n"
    if conversation:
        prompt += "Conversation so far:\n" + "\n".join(conversation) + "\n\n"
    prompt += "Question: " + query + "\n\nAnswer:"
    return prompt


def build_no_context_prompt(query: str, history: Optional[List[Dict[str, str]]] = None) -> str:
    conversation = []
    if history:
        for turn in history[-6:]:
            conversation.append(f"User: {turn['user']}")
            conversation.append(f"Assistant: {turn['assistant']}")
            
    prompt = SYSTEM_PROMPT_TEMPLATE.format(rag_context="No context available.") + "\n\n"
    if conversation:
        prompt += "Conversation so far:\n" + "\n".join(conversation) + "\n\n"
    prompt += "Question: " + query + "\n\nAnswer:"
    return prompt


def _load_documents() -> List[Any]:
    all_documents = []
    use_visual = os.getenv("USE_VISUAL_PARSING", "false").lower() in ("true", "1", "yes")

    if os.path.exists(LOCAL_DATA_DIR):
        if os.path.isdir(LOCAL_DATA_DIR):
            if use_visual:
                print(f"Visual PDF Ingestion enabled. Scanning directory '{LOCAL_DATA_DIR}' for PDF files...")
                from visual_pdf_loader import VisualPDFLoader
                for file_name in os.listdir(LOCAL_DATA_DIR):
                    if file_name.lower().endswith(".pdf"):
                        pdf_path = os.path.join(LOCAL_DATA_DIR, file_name)
                        try:
                            loader = VisualPDFLoader(pdf_path)
                            all_documents.extend(loader.load())
                        except Exception as e:
                            print(f"Warning: Visual PDF parsing failed for {pdf_path}: {e}. Falling back to standard loader.")
                            try:
                                from langchain_community.document_loaders import PyPDFLoader
                                all_documents.extend(PyPDFLoader(pdf_path).load())
                            except Exception as e2:
                                print(f"Warning: standard PDF fallback loader also failed: {e2}")
            else:
                try:
                    all_documents.extend(PyPDFDirectoryLoader(LOCAL_DATA_DIR).load())
                except ImportError as e:
                    print(f"Warning: PDF loading is unavailable: {e}")
                    print("Install pypdf or continue with text/web sources only.")

            text_loader = DirectoryLoader(LOCAL_DATA_DIR, glob="**/*.txt", loader_cls=TextLoader)
            all_documents.extend(text_loader.load())
        elif os.path.isfile(LOCAL_DATA_DIR):
            ext = os.path.splitext(LOCAL_DATA_DIR)[1].lower()
            if ext == ".pdf":
                use_fallback = True
                if use_visual:
                    try:
                        from visual_pdf_loader import VisualPDFLoader
                        loader = VisualPDFLoader(LOCAL_DATA_DIR)
                        all_documents.extend(loader.load())
                        use_fallback = False
                    except Exception as e:
                        print(f"Warning: Visual PDF parsing failed for {LOCAL_DATA_DIR}: {e}. Falling back to standard loader.")
                
                if use_fallback:
                    try:
                        from langchain_community.document_loaders import PyPDFLoader
                    except Exception:
                        PyPDFLoader = None

                    if PyPDFLoader is not None:
                        try:
                            all_documents.extend(PyPDFLoader(LOCAL_DATA_DIR).load())
                        except ImportError as e:
                            print(f"Warning: PDF loading is unavailable: {e}")
                            print("Install pypdf or continue with text/web sources only.")
                    else:
                        raise RuntimeError("PyPDFLoader not available; install the appropriate loader or provide a directory of PDFs")
            elif ext in (".txt", ".md"):
                all_documents.extend(TextLoader(LOCAL_DATA_DIR).load())
            else:
                raise RuntimeError(f"Unsupported file type for LOCAL_DATA_DIR: {ext}")

    if TARGET_URLS:
        try:
            web_loader = WebBaseLoader(web_paths=TARGET_URLS, raise_for_status=False)
            all_documents.extend(web_loader.load())
        except Exception as e:
            print(f"Warning: failed to load target URLs: {e}")
            print("Continuing with local documents only.")

    if not all_documents:
        raise RuntimeError("No documents were loaded. Check LOCAL_DATA_DIR and TARGET_URLS.")

    return all_documents


def build_vector_database() -> int:
    all_documents = _load_documents()
    chunks = chunk_documents(all_documents)
    docs_texts = [doc.page_content for doc in chunks]
    metadatas = [dict(doc.metadata, chunk_index=i) for i, doc in enumerate(chunks)]
    ids = [f"{m.get('source', 'doc')}|{i}" for i, m in enumerate(metadatas)]

    client = chroma_cloud.get_client()
    base_name = os.getenv("CHROMA_DATABASE") or "Jokaan_RAG_exe"
    dense_fn = chroma_cloud.get_dense_embedding_function(task=None)
    sparse_fn = chroma_cloud.get_sparse_embedding_function()
    collection_name = chroma_cloud.get_collection_name(base_name, dense_fn)
    coll = chroma_cloud.get_or_create_collection(
        client,
        collection_name,
        metadata={"source": "local_data"},
        dense_embedding_fn=dense_fn,
        sparse_embedding_fn=sparse_fn,
    )
    embeddings = dense_fn(docs_texts)
    chroma_cloud.upsert_documents(coll, ids=ids, documents=docs_texts, metadatas=metadatas, embeddings=embeddings)
    return len(ids)


def initialize_pipeline() -> None:
    global _llm_with_search, _llm_without_search, _retriever
    with _pipeline_lock:
        if _llm_without_search is not None and _retriever is not None:
            return

        _llm_with_search = get_llm(enable_search=True)
        _llm_without_search = get_llm(enable_search=False)
        dense_fn = chroma_cloud.get_dense_embedding_function(task=None)
        sparse_fn = chroma_cloud.get_sparse_embedding_function()
        base_name = os.getenv("CHROMA_DATABASE") or "Jokaan_RAG_exe"
        collection_name = chroma_cloud.get_collection_name(base_name, dense_fn)
        client = chroma_cloud.get_client()

        if not chroma_cloud.collection_exists(client, collection_name) or chroma_cloud.collection_document_count(client, collection_name) == 0:
            build_vector_database()

        _retriever = chroma_cloud.get_retriever(
            base_name,
            embedding_fn=dense_fn,
            k=DEFAULT_TOP_K,
            sparse_embedding_fn=sparse_fn,
            dedupe_by_source=False,
        )


def answer_query(query: str, history: Optional[List[Dict[str, str]]] = None, top_k: int = DEFAULT_TOP_K) -> Dict[str, Any]:
    if not query:
        raise ValueError("Query must not be empty.")

    initialize_pipeline()
    if _retriever is None or _llm_without_search is None or _llm_with_search is None:
        raise RuntimeError("RAG pipeline is not initialized.")

    # Retrieve candidate documents (fetch 15 if rerank is enabled)
    reranker = get_reranker()
    candidate_k = 15 if reranker is not None else top_k

    original_k = _retriever.k
    _retriever.k = candidate_k
    try:
        docs = _retriever.get_relevant_documents(query)
    finally:
        _retriever.k = original_k

    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    if provider == "ollama":
        top_k = min(top_k, 2)

    # Perform re-ranking
    if docs and reranker is not None:
        try:
            from flashrank import RerankRequest
            passages = [
                {"id": idx, "text": doc.page_content, "meta": doc.metadata}
                for idx, doc in enumerate(docs)
            ]
            rerank_request = RerankRequest(query=query, passages=passages)
            ranked_results = reranker.rerank(rerank_request)
            docs = [docs[item["id"]] for item in ranked_results[:top_k]]
        except Exception as e:
            print(f"Warning: re-ranking failed: {e}. Falling back to default retrieval.")
            docs = docs[:top_k]
    else:
        docs = docs[:top_k]

    if docs:
        prompt_text = build_prompt(query, docs, history)
        llm = _llm_without_search
    else:
        prompt_text = build_no_context_prompt(query, history)
        llm = _llm_with_search

    try:
        try:
            response = llm.invoke(prompt_text)
        except AttributeError:
            response = llm(prompt_text)
    except Exception as e:
        provider = os.getenv("LLM_PROVIDER", "gemini").lower()
        if provider == "gemini":
            print(f"Gemini error: {e}. Falling back to local Ollama model...")
            local_llm = OllamaChatModel(
                model_name=os.getenv("OLLAMA_MODEL", "llama3.2"),
                temperature=0.1
            )
            response = local_llm.invoke(prompt_text)
        else:
            raise e

    answer_text = extract_response_text(response).strip()
    return {
        "query": query,
        "answer": answer_text,
        "retrieved_count": len(docs),
        "context": [
            {
                "source": doc.metadata.get("source") or doc.metadata.get("_id") or "unknown",
                "chunk_index": doc.metadata.get("chunk_index"),
                "text": doc.page_content.strip(),
            }
            for doc in docs
        ],
    }
