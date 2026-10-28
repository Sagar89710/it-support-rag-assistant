"""Enterprise IT support assistant using LangChain, FAISS, and LangGraph."""
from __future__ import annotations

import argparse
import hashlib
import os
import re
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph

BASE_DIR = Path(__file__).parent
KNOWLEDGE_BASE = BASE_DIR / "data" / "support_knowledge_base.md"
STOP_WORDS = {"a", "an", "and", "be", "can", "company", "do", "for", "how", "i", "in", "is", "it", "my", "of", "on", "should", "the", "to", "what", "when", "with"}


# LangGraph passes this shared state from one workflow node to the next.
class AssistantState(TypedDict, total=False):
    question: str
    documents: list[Document]
    answer: str
    sources: list[str]
    validation: str


class LocalKeywordEmbeddings(Embeddings):
    """Small deterministic offline embedding used for the local demonstration."""

    dimensions = 256

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            index = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16) % self.dimensions
            vector[index] += 1.0
        magnitude = sum(value * value for value in vector) ** 0.5 or 1.0
        return [value / magnitude for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def load_documents() -> list[Document]:
    """Load the knowledge base and preserve section headings as metadata."""
    text = KNOWLEDGE_BASE.read_text(encoding="utf-8")
    sections = []
    current_title = "General"
    current_lines: list[str] = []

    for line in text.splitlines():
        if line.startswith("## "):
            if current_lines:
                sections.append(
                    Document(
                        page_content="\n".join(current_lines).strip(),
                        metadata={"source": KNOWLEDGE_BASE.name, "section": current_title},
                    )
                )
            current_title = line.removeprefix("## ").strip()
            current_lines = []
        elif not line.startswith("# "):
            current_lines.append(line)

    if current_lines:
        sections.append(
            Document(
                page_content="\n".join(current_lines).strip(),
                metadata={"source": KNOWLEDGE_BASE.name, "section": current_title},
            )
        )
    return sections


def build_vector_store(use_openai: bool = False) -> FAISS:
    documents = load_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    chunks = splitter.split_documents(documents)

    if use_openai:
        embeddings = OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        )
    else:
        embeddings = LocalKeywordEmbeddings()
    return FAISS.from_documents(chunks, embeddings)


def format_sources(documents: list[Document]) -> list[str]:
    return [
        f"{document.metadata.get('source')} [{document.metadata.get('section')}]"
        for document in documents
    ]


def rerank_documents(question: str, documents: list[Document]) -> list[Document]:
    question_terms = {
        term for term in re.findall(r"[a-z0-9]+", question.lower()) if term not in STOP_WORDS
    }

    def score(document: Document) -> int:
        searchable_text = f"{document.metadata.get('section', '')} {document.page_content}".lower()
        document_terms = set(re.findall(r"[a-z0-9]+", searchable_text))
        return len(question_terms & document_terms)

    ranked_documents = sorted(documents, key=score, reverse=True)
    if not ranked_documents or score(ranked_documents[0]) < 2:
        return []
    return ranked_documents


def local_answer(question: str, documents: list[Document]) -> str:
    """Provide a deterministic demo answer without an external LLM."""
    if not documents:
        return "I could not find an answer in the support knowledge base."
    content = documents[0].page_content.replace("**", "")
    return f"Based on the support knowledge base: {content}"


def create_graph(vector_store: FAISS, use_openai: bool = False):
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini")) if use_openai else None

    # Each function below is one step in the LangGraph workflow.
    def retrieve(state: AssistantState) -> AssistantState:
        # Find document chunks that are relevant to the user's question.
        candidates = vector_store.similarity_search(state["question"], k=4)
        documents = rerank_documents(state["question"], candidates)[:2]
        return {"documents": documents, "sources": format_sources(documents)}

    def generate(state: AssistantState) -> AssistantState:
        # Create an answer from the retrieved context, using an LLM when enabled.
        documents = state.get("documents", [])
        context = "\n\n".join(document.page_content for document in documents)
        if llm:
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "Answer only from the supplied context. If it is not present, say you do not know. Keep the answer concise."),
                    ("human", "Context:\n{context}\n\nQuestion: {question}"),
                ]
            )
            answer = (prompt | llm).invoke({"context": context, "question": state["question"]}).content
        else:
            answer = local_answer(state["question"], documents)
        return {"answer": str(answer)}

    def validate(state: AssistantState) -> AssistantState:
        # Only complete the workflow when an answer has supporting sources.
        answer = state.get("answer", "")
        sources = state.get("sources", [])
        if answer and sources and "could not find" not in answer.lower():
            return {"validation": "passed"}
        return {"validation": "fallback"}

    def route_after_validation(state: AssistantState) -> str:
        # This decision is why LangGraph is useful: it can route to fallback conditionally.
        return "complete" if state.get("validation") == "passed" else "fallback"

    def fallback(state: AssistantState) -> AssistantState:
        # Avoid inventing an answer when the knowledge base has no reliable match.
        return {
            "answer": "I could not find a reliable answer in the support knowledge base. Please raise a support ticket with the relevant details.",
            "validation": "fallback",
        }

    # Build the workflow: retrieve -> generate -> validate -> complete or fallback.
    graph = StateGraph(AssistantState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("validate", validate)
    graph.add_node("fallback", fallback)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "validate")
    graph.add_conditional_edges("validate", route_after_validation, {"complete": END, "fallback": "fallback"})
    graph.add_edge("fallback", END)
    return graph.compile()


def answer_question(question: str, use_openai: bool = False) -> AssistantState:
    """Run one question through the complete retrieval and response workflow."""
    vector_store = build_vector_store(use_openai=use_openai)
    graph = create_graph(vector_store, use_openai=use_openai)
    return graph.invoke({"question": question})


def main() -> None:
    load_dotenv(BASE_DIR / ".env")
    parser = argparse.ArgumentParser(description="Run the RAG and LangGraph IT support assistant.")
    parser.add_argument("question", help="Question to ask the support knowledge base")
    parser.add_argument("--openai", action="store_true", help="Use OpenAI embeddings and chat completion")
    args = parser.parse_args()

    if args.openai and not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required when using --openai.")

    result = answer_question(args.question, use_openai=args.openai)
    print(f"Answer: {result['answer']}")
    print(f"Sources: {', '.join(result.get('sources', [])) or 'None'}")
    print(f"Validation: {result.get('validation', 'unknown')}")


if __name__ == "__main__":
    main()
