import os
import hashlib
import json
import base64
import logging
import time
import random
import fitz  # PyMuPDF
from typing import List
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

class VisualPDFLoader:
    """A document loader that converts PDF pages to images and uses Gemini's
    multimodal capabilities to generate a rich markdown representation of each page.
    It includes a file-hashing cache to avoid duplicate LLM invocations.
    """
    def __init__(self, file_path: str, cache_dir: str = None, force_rebuild: bool = False):
        self.file_path = file_path
        self.cache_dir = cache_dir or os.getenv("VISUAL_PARSING_CACHE_DIR", "./.cache/visual_pdf")
        self.force_rebuild = force_rebuild
        
        # Load API key and config
        load_dotenv(override=True)
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.model_name = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")

    def _get_file_hash(self) -> str:
        """Calculate the SHA-256 hash of the PDF file."""
        sha256 = hashlib.sha256()
        with open(self.file_path, "rb") as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _save_incremental_cache(self, cache_file: str, cached_pages: dict, file_hash: str):
        """Save current successfully parsed pages to cache JSON file atomically."""
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            pages_list = []
            for p_num in sorted(cached_pages.keys()):
                pages_list.append({
                    "page_num": p_num,
                    "content": cached_pages[p_num]
                })
            
            cache_data = {
                "file_path": self.file_path,
                "file_hash": file_hash,
                "pages": pages_list
            }
            
            # Atomic write using a temp file
            temp_cache_file = cache_file + ".tmp"
            with open(temp_cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            os.replace(temp_cache_file, cache_file)
        except Exception as e:
            logger.warning(f"Failed to write incremental cache: {e}")

    def load(self) -> List[Document]:
        """Convert pages to images and parse using Gemini with retry mechanisms."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"PDF file not found: {self.file_path}")

        file_hash = self._get_file_hash()
        cache_file = os.path.join(self.cache_dir, f"{file_hash}.json")

        # 1. Load existing parsed pages from cache if available
        cached_pages = {}
        if not self.force_rebuild and os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)
                
                raw_pages = cache_data.get("pages", [])
                if isinstance(raw_pages, list):
                    for item in raw_pages:
                        cached_pages[int(item.get("page_num"))] = item.get("content", "")
                elif isinstance(raw_pages, dict):
                    for k, v in raw_pages.items():
                        cached_pages[int(k)] = str(v)
            except Exception as e:
                logger.warning(f"Failed to read existing cache file {cache_file}: {e}")

        # 2. Check if all pages are already parsed
        doc = fitz.open(self.file_path)
        total_pages = len(doc)
        
        all_pages_cached = all(p in cached_pages for p in range(1, total_pages + 1))
        if not self.force_rebuild and all_pages_cached:
            print(f"All {total_pages} pages loaded from visual cache.")
            documents = []
            for p in range(1, total_pages + 1):
                metadata = {
                    "source": self.file_path,
                    "page": p,
                    "parser": "gemini-visual-cache"
                }
                documents.append(Document(page_content=cached_pages[p], metadata=metadata))
            return documents

        # 3. Parse missing pages
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY is required to run visual PDF parsing. Please check your .env file.")

        print(f"Starting visual PDF parsing for '{os.path.basename(self.file_path)}' using Gemini...")
        print(f"  (Loaded {len(cached_pages)}/{total_pages} pages from cache. Parsing remaining pages...)")
        
        # Initialize Gemini LLM via LangChain
        llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=0.1,
            google_api_key=self.api_key
        )

        prompt = (
            "Extract all text, product names, tables, and specifications from this page. "
            "Reconstruct any tables as Markdown tables. "
            "Describe any product images, diagrams, or illustrations in detail. "
            "Format the output in clean Markdown. Output only the parsed content."
        )

        MAX_RETRIES = 5
        INITIAL_BACKOFF = 3.0

        documents = []
        for page_idx in range(total_pages):
            page_num = page_idx + 1
            
            # Check if this page is already in the cache
            if not self.force_rebuild and page_num in cached_pages:
                print(f"  -> Page {page_num}/{total_pages} loaded from cache.")
                content = cached_pages[page_num]
                metadata = {
                    "source": self.file_path,
                    "page": page_num,
                    "parser": "gemini-visual-cache"
                }
                documents.append(Document(page_content=content, metadata=metadata))
                continue

            print(f"  -> Visually parsing page {page_num}/{total_pages} with Gemini...")
            page = doc.load_page(page_idx)
            
            # Render page to image
            pix = page.get_pixmap(dpi=150)
            img_bytes = pix.tobytes("jpeg")
            base64_image = base64.b64encode(img_bytes).decode("utf-8")

            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            )

            success = False
            retries = 0
            backoff = INITIAL_BACKOFF
            content = ""

            while not success and retries < MAX_RETRIES:
                try:
                    response = llm.invoke([message])
                    content = response.content.strip()
                    success = True
                except Exception as e:
                    retries += 1
                    if retries >= MAX_RETRIES:
                        logger.error(f"Failed parsing page {page_num} after {MAX_RETRIES} retries: {e}")
                        print(f"  [ERROR] Page {page_num} failed completely after {MAX_RETRIES} retries: {e}")
                        content = f"Error parsing page {page_num}: {e}"
                        break
                    
                    sleep_time = backoff + random.uniform(0.5, 1.5)
                    print(f"    [Retry {retries}/{MAX_RETRIES}] Error calling Gemini: {e}. Retrying in {sleep_time:.2f}s...")
                    time.sleep(sleep_time)
                    backoff *= 2.0  # double the backoff

            metadata = {
                "source": self.file_path,
                "page": page_num,
                "parser": "gemini-visual-llm" if success else "gemini-visual-error"
            }
            documents.append(Document(page_content=content, metadata=metadata))

            if success:
                cached_pages[page_num] = content
                self._save_incremental_cache(cache_file, cached_pages, file_hash)
                # Polite rate limit delay between API queries
                time.sleep(2.0)
            else:
                # Extra cooldown if error occurred
                time.sleep(5.0)

        return documents
