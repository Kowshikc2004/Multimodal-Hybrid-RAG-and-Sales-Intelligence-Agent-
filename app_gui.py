import os
import threading
import requests
import customtkinter as ctk
from tkinter import messagebox
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

from langchain_google_genai import ChatGoogleGenerativeAI
import chroma_cloud
import rag_service

DEFAULT_MODEL = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")


class RAGChatbotGUI:
    def __init__(self, root):
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        self.root = root
        self.root.title("Jokaan RAG Assistant")
        self.root.geometry("980x720")
        self.root.minsize(900, 650)

        self.llm = None
        self.retriever = None
        self.use_api = False
        self.api_url = os.getenv("RAG_API_URL") or os.getenv("API_URL")
        self.chat_history = []
        self.message_count = 0

        self._build_ui()
        self._initialize_pipeline_async()

    def _build_ui(self):
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        self.sidebar = ctk.CTkFrame(self.root, width=240, corner_radius=20)
        self.sidebar.grid(row=0, column=0, sticky="nsw", padx=(16, 8), pady=16)
        self.sidebar.grid_rowconfigure((0, 1, 2, 3, 4, 5), weight=0)
        self.sidebar.grid_rowconfigure(6, weight=1)

        self.main_panel = ctk.CTkFrame(self.root, corner_radius=20)
        self.main_panel.grid(row=0, column=1, sticky="nsew", padx=(8, 16), pady=16)
        self.main_panel.grid_rowconfigure(0, weight=1)
        self.main_panel.grid_rowconfigure(1, weight=0)
        self.main_panel.grid_columnconfigure(0, weight=1)

        self.sidebar_title = ctk.CTkLabel(self.sidebar, text="Jokaan RAG", font=ctk.CTkFont(size=20, weight="bold"))
        self.sidebar_title.grid(row=0, column=0, pady=(20, 10), padx=16, sticky="w")

        self.db_status_label = ctk.CTkLabel(self.sidebar, text="Vector DB: initializing...", font=ctk.CTkFont(size=13), anchor="w")
        self.db_status_label.grid(row=1, column=0, pady=(0, 10), padx=16, sticky="w")

        self.status_label = ctk.CTkLabel(self.sidebar, text="Status: starting", font=ctk.CTkFont(size=12), anchor="w")
        self.status_label.grid(row=2, column=0, pady=(0, 20), padx=16, sticky="w")

        self.clear_button = ctk.CTkButton(self.sidebar, text="Clear Chat", fg_color="#3083dc", hover_color="#1f6bb8", command=self.clear_chat)
        self.clear_button.grid(row=3, column=0, pady=(0, 8), padx=16, sticky="ew")

        self.mode_button = ctk.CTkButton(self.sidebar, text="Toggle Light/Dark", command=self.toggle_appearance_mode)
        self.mode_button.grid(row=4, column=0, pady=(0, 8), padx=16, sticky="ew")

        self.instructions_label = ctk.CTkLabel(
            self.sidebar,
            text="Ask Jokaan questions in plain English and the assistant will answer using the document knowledge base.",
            wraplength=200,
            justify="left",
            font=ctk.CTkFont(size=12),
            anchor="w",
        )
        self.instructions_label.grid(row=5, column=0, pady=(8, 0), padx=16, sticky="w")

        self.chat_frame = ctk.CTkFrame(self.main_panel, corner_radius=20)
        self.chat_frame.grid(row=0, column=0, sticky="nsew", padx=16, pady=16)
        self.chat_frame.grid_rowconfigure(0, weight=1)
        self.chat_frame.grid_columnconfigure(0, weight=1)

        self.chat_container = ctk.CTkScrollableFrame(self.chat_frame, corner_radius=20, fg_color="#1f2024")
        self.chat_container.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.chat_container.grid_columnconfigure(0, weight=1)

        self.input_panel = ctk.CTkFrame(self.main_panel, corner_radius=20)
        self.input_panel.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 16))
        self.input_panel.grid_columnconfigure(0, weight=1)
        self.input_panel.grid_columnconfigure(1, weight=0)

        self.input_box = ctk.CTkTextbox(self.input_panel, height=130, corner_radius=16, font=ctk.CTkFont(size=13))
        self.input_box.grid(row=0, column=0, padx=(16, 8), pady=16, sticky="nsew")
        self.input_box.bind("<Return>", self._on_enter_pressed)
        self.input_box.bind("<Shift-Return>", self._insert_newline)

        self.send_button = ctk.CTkButton(self.input_panel, text="Send", width=120, fg_color="#4285F4", hover_color="#3367D6", command=self.send_message)
        self.send_button.grid(row=0, column=1, padx=(8, 16), pady=16, sticky="n")

        self.typing_label = ctk.CTkLabel(self.main_panel, text="", font=ctk.CTkFont(size=12), anchor="w")
        self.typing_label.grid(row=2, column=0, padx=24, pady=(0, 8), sticky="w")

    def toggle_appearance_mode(self):
        current_mode = ctk.get_appearance_mode()
        ctk.set_appearance_mode("Light" if current_mode == "Dark" else "Dark")

    def _initialize_pipeline_async(self):
        self.send_button.configure(state="disabled")
        self.status_label.configure(text="Status: loading models...")
        threading.Thread(target=self.init_pipeline, daemon=True).start()

    def init_pipeline(self):
        try:
            # If an external RAG API is configured, prefer using it by calling its health endpoint.
            if self.api_url:
                try:
                    health = requests.get(f"{self.api_url.rstrip('/')}/health", timeout=6)
                    if health.status_code == 200:
                        # Do a quick probe POST to ensure /query works before switching to remote mode
                        try:
                            probe = requests.post(f"{self.api_url.rstrip('/')}/query", json={"query": "ping", "include_context": False, "top_k": 1}, timeout=8)
                            if probe.status_code == 200:
                                self.use_api = True
                                self.root.after(0, lambda: self._update_ready_state())
                                return
                        except Exception:
                            # health passed but probe failed; log and fall back
                            self._update_status("remote API probe failed; falling back to local pipeline")
                except Exception:
                    # Fall back to local pipeline initialization
                    self._update_status("remote API not reachable; falling back to local pipeline")

            # Local pipeline (existing behavior)
            self.llm = rag_service.get_llm()
            dense_fn = chroma_cloud.get_dense_embedding_function(task=None)
            sparse_fn = chroma_cloud.get_sparse_embedding_function()
            base_name = os.getenv("CHROMA_DATABASE") or "Jokaan_RAG_exe"
            collection_name = chroma_cloud.get_collection_name(base_name, dense_fn)
            client = chroma_cloud.get_client()

            if not chroma_cloud.collection_exists(client, collection_name) or chroma_cloud.collection_document_count(client, collection_name) == 0:
                self.root.after(0, lambda: self._update_status("Building vector database..."))
                rag_service.build_vector_database()

            self.retriever = chroma_cloud.get_retriever(
                base_name,
                embedding_fn=dense_fn,
                k=3,
                sparse_embedding_fn=sparse_fn,
                dedupe_by_source=False,
            )

            self.root.after(0, lambda: self._update_ready_state())
        except Exception as exc:
            self.root.after(0, lambda: self._show_init_error(exc))

    def _update_ready_state(self):
        self.db_status_label.configure(text="Vector DB: ready", text_color="#3DDC84")
        self.status_label.configure(text="Status: system ready")
        self.send_button.configure(state="normal")
        self.add_system_message("Hello! I am ready to answer questions about Jokaan.")

    def _update_status(self, message: str):
        self.status_label.configure(text=f"Status: {message}")

    def _show_init_error(self, error):
        self.send_button.configure(state="disabled")
        self.db_status_label.configure(text="Vector DB: error", text_color="#FF6B6B")
        self.status_label.configure(text="Status: initialization failed")
        messagebox.showerror("Initialization Error", str(error))

    def add_system_message(self, text: str):
        self.add_chat_bubble("System", text, system=True)

    def add_chat_bubble(self, sender: str, text: str, system: bool = False):
        bubble_color = "#2f3943" if sender == "Bot" else "#4285F4"
        text_color = "#f1f5f9" if sender == "Bot" else "#ffffff"
        anchor = "w" if sender == "Bot" else "e"
        padx = (12, 80) if sender == "Bot" else (80, 12)
        fg = "#28323d" if sender == "Bot" else "#3e7ef0"

        bubble_frame = ctk.CTkFrame(self.chat_container, fg_color=fg, corner_radius=18, border_width=0)
        bubble_frame.grid_columnconfigure(0, weight=1)

        label = ctk.CTkLabel(
            bubble_frame,
            text=text,
            wraplength=520,
            justify="left",
            font=ctk.CTkFont(size=13),
            text_color=text_color,
            anchor="w",
        )
        label.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        if sender == "Bot" or system:
            bubble_frame.grid(row=self.message_count, column=0, sticky=anchor, padx=padx, pady=(8, 4))
        else:
            bubble_frame.grid(row=self.message_count, column=0, sticky=anchor, padx=padx, pady=(8, 4))

        self.message_count += 1
        self.chat_container.update_idletasks()
        try:
            self.chat_container.yview_moveto(1.0)
        except Exception:
            pass

    def _on_enter_pressed(self, event):
        if event.state & 0x0001:
            return
        self.send_message()
        return "break"

    def _insert_newline(self, event):
        self.input_box.insert("insert", "\n")
        return "break"

    def clear_chat(self):
        for child in self.chat_container.winfo_children():
            child.destroy()
        self.chat_history = []
        self.message_count = 0
        self._update_status("chat cleared")

    def send_message(self):
        query = self.input_box.get("0.0", "end").strip()
        if not query or self.llm is None or self.retriever is None:
            return

        self.add_chat_bubble("You", query)
        self.chat_history.append({"user": query, "assistant": ""})
        self.input_box.delete("0.0", "end")
        self.send_button.configure(state="disabled")
        self.input_box.configure(state="disabled")
        self.typing_label.configure(text="Typing...", text_color="#3DDC84")
        self._update_status("gathering context")

        threading.Thread(target=self.process_query, args=(query,), daemon=True).start()

    def process_query(self, query):
        try:
            if self.use_api:
                payload = {"query": query, "include_context": True, "top_k": 3}
                resp = requests.post(f"{self.api_url.rstrip('/')}/query", json=payload, timeout=20)
                resp.raise_for_status()
                data = resp.json()
                answer_text = data.get("answer", "")
                self.chat_history[-1]["assistant"] = answer_text
                self.root.after(0, self._display_response, answer_text)
                return

            docs = self.retriever.get_relevant_documents(query)
            provider = os.getenv("LLM_PROVIDER", "gemini").lower()
            if provider == "ollama":
                docs = docs[:2]
            if docs:
                prompt_text = rag_service.build_prompt(query, docs, self.chat_history[:-1])
            else:
                prompt_text = rag_service.build_no_context_prompt(query, self.chat_history[:-1])

            llm = rag_service.get_llm_for_docs(docs)
            try:
                try:
                    response = llm.invoke(prompt_text)
                except AttributeError:
                    response = llm(prompt_text)
            except Exception as e:
                provider = os.getenv("LLM_PROVIDER", "gemini").lower()
                if provider == "gemini":
                    print(f"Gemini error: {e}. Falling back to local Ollama model...")
                    local_llm = rag_service.OllamaChatModel(
                        model_name=os.getenv("OLLAMA_MODEL", "llama3.2"),
                        temperature=0.1
                    )
                    response = local_llm.invoke(prompt_text)
                else:
                    raise e

            response_text = rag_service.extract_response_text(response).strip()
            self.chat_history[-1]["assistant"] = response_text
            self.root.after(0, self._display_response, response_text)
        except Exception as exc:
            self.root.after(0, self._handle_query_error, exc)

    def _display_response(self, response_text):
        self.add_chat_bubble("Bot", response_text)
        self.typing_label.configure(text="")
        self._update_status("system ready")
        self.send_button.configure(state="normal")
        self.input_box.configure(state="normal")

    def _handle_query_error(self, error):
        self.typing_label.configure(text="")
        self._update_status("error occurred")
        messagebox.showerror("Query Error", str(error))
        self.send_button.configure(state="normal")
        self.input_box.configure(state="normal")


if __name__ == "__main__":
    app = ctk.CTk()
    RAGChatbotGUI(app)
    app.mainloop()
