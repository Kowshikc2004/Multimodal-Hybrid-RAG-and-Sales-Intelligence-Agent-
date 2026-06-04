# Implementation Plan - Multimodal RAG (Visual PDF Parsing)

This plan outlines the implementation of **Multimodal RAG with Visual PDF Ingestion** for Jokaan. Currently, PDFs are parsed using standard text extraction (which loses layout, tables, diagrams, and visual callouts). By converting PDF pages to images and using Gemini's native visual/multimodal capabilities, we will index high-fidelity page descriptions and structured markdown representations of the tables and illustrations.

---

## User Review Required

> [!IMPORTANT]
> - **API Cost / Rate Limits**: Since the PDF is processed page-by-page, indexing a 21-page PDF requires 21 calls to Gemini 2.5 Flash. We will implement a file-hash-based cache (`.cache/visual_pdf/`) so that subsequent rebuilds or migrations are instant and do not incur additional API costs or rate limit issues.
> - **PyMuPDF Dependency**: PyMuPDF (`fitz`) is already installed in your virtual environment and will be used to render the PDF pages.

---

## Proposed Changes

### Configuration

#### [MODIFY] [.env](file:///Users/kowshikch/Desktop/LangChain%20copy/Jokaan%20RAG%20exe/.env)
- Add environment variables to control visual PDF parsing:
  ```bash
  # Enable Gemini Visual PDF Ingestion
  USE_VISUAL_PARSING=true
  # Cache directory for parsed pages
  VISUAL_PARSING_CACHE_DIR=./.cache/visual_pdf
  ```

---

### Core Components

#### [NEW] [visual_pdf_loader.py](file:///Users/kowshikch/Desktop/LangChain%20copy/Jokaan%20RAG%20exe/visual_pdf_loader.py)
Create a new utility class to handle the visual PDF parsing pipeline.
- **`VisualPDFLoader`**:
  - Computes a SHA-256 hash of the PDF file to check if a cached version exists under `VISUAL_PARSING_CACHE_DIR`.
  - If a cache exists, reads the markdown content for each page from the cache.
  - If not, opens the PDF using `fitz` (PyMuPDF).
  - Renders each page as a JPEG image (DPI=150).
  - Encodes images to base64.
  - Sends the image to Gemini 2.5 Flash using `ChatGoogleGenerativeAI` with a specialized parsing prompt:
    ```
    Extract all text, product names, tables, and specifications from this page.
    Reconstruct any tables as Markdown tables.
    Describe any product images, diagrams, or illustrations in detail.
    Format the output in clean Markdown.
    ```
  - Caches the generated page contents to a JSON file.
  - Returns a list of LangChain `Document` objects with appropriate metadata (`source`, `page`).

---

### Ingestion Service

#### [MODIFY] [rag_service.py](file:///Users/kowshikch/Desktop/LangChain%20copy/Jokaan%20RAG%20exe/rag_service.py)
Integrate the `VisualPDFLoader` into the document loading step.
- Update `_load_documents()`:
  - If `USE_VISUAL_PARSING` is enabled and a PDF file/directory is loaded:
    - Invoke `VisualPDFLoader.load(pdf_path)` to ingest files visually instead of `PyPDFLoader` or `PyPDFDirectoryLoader`.
  - Fall back to standard `PyPDFLoader` if visual parsing is disabled or fails.

---

### Dependencies

#### [MODIFY] [requirements.txt](file:///Users/kowshikch/Desktop/LangChain%20copy/Jokaan%20RAG%20exe/requirements.txt)
- Document the `pymupdf` library by appending it to the requirements list.

---

## Verification Plan

### Automated Tests
1. **Visual Loader Test**:
   Run a standalone command to verify document loading and caching:
   ```bash
   python -c "from visual_pdf_loader import VisualPDFLoader; docs = VisualPDFLoader('JOKAAN-PORTFOLIO NOV2025-v2.pdf').load(); print(f'Loaded {len(docs)} documents. Page 1 text length: {len(docs[0].page_content)}')"
   ```
2. **Re-build Database Test**:
   Run `migrate_to_chroma.py` to ensure that database rebuild reads from cache and completes successfully:
   ```bash
   python migrate_to_chroma.py
   ```

### Manual Verification
- In the Jokaan RAG GUI or API, query about specifications, product diagrams, or tables (e.g. "What is the battery backup of the DVT prevention pump?").
- Verify that the returned context contains structured markdown tables and details that were previously lost or unstructured.
