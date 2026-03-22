"""
ChromaDB-backed memory store for Annabeth.

Collections:
- conversations: Summarized past conversation turns
- facts: Learned facts about users (names, preferences, relationships)
- self_notes: Annabeth's notes about her own behavior and performance
"""
import time
from pathlib import Path
from typing import Optional

import chromadb

# Singleton instance
_store: Optional["MemoryStore"] = None

# ChromaDB on C: NVMe for fast random reads
CHROMADB_PATH = Path(r"C:\annabeth_data\chromadb")


def get_memory_store() -> "MemoryStore":
    """Get or create the singleton MemoryStore."""
    global _store
    if _store is None:
        _store = MemoryStore()
    return _store


class MemoryStore:
    """Wrapper around ChromaDB for Annabeth's long-term memory."""

    def __init__(self, persist_dir: Path = CHROMADB_PATH):
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_dir))

        # Create/get collections
        self.conversations = self._client.get_or_create_collection(
            name="conversations",
            metadata={"hnsw:space": "cosine"},
        )
        self.facts = self._client.get_or_create_collection(
            name="facts",
            metadata={"hnsw:space": "cosine"},
        )
        self.self_notes = self._client.get_or_create_collection(
            name="self_notes",
            metadata={"hnsw:space": "cosine"},
        )
        print(f"[Memory] ChromaDB loaded from {persist_dir}")
        print(f"[Memory] conversations={self.conversations.count()}, "
              f"facts={self.facts.count()}, self_notes={self.self_notes.count()}")

    # ----- Conversations -----

    def add_conversation(self, summary: str, speaker: str = "Unknown",
                         metadata: Optional[dict] = None) -> str:
        """Store a conversation summary for later recall."""
        doc_id = f"conv_{int(time.time() * 1000)}"
        meta = {
            "speaker": speaker,
            "timestamp": time.time(),
            "type": "conversation",
        }
        if metadata:
            meta.update(metadata)
        self.conversations.add(
            documents=[summary],
            metadatas=[meta],
            ids=[doc_id],
        )
        return doc_id

    def recall_conversations(self, query: str, n_results: int = 3,
                             speaker: Optional[str] = None) -> list[dict]:
        """Find past conversations relevant to a query."""
        where = {"speaker": speaker} if speaker else None
        results = self.conversations.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
        )
        return self._unpack(results)

    # ----- Facts -----

    def add_fact(self, fact: str, subject: str = "general",
                 speaker: str = "Unknown") -> str:
        """Store a learned fact (e.g., 'Dad likes sci-fi movies')."""
        doc_id = f"fact_{int(time.time() * 1000)}"
        self.facts.add(
            documents=[fact],
            metadatas=[{
                "subject": subject,
                "speaker": speaker,
                "timestamp": time.time(),
                "type": "fact",
            }],
            ids=[doc_id],
        )
        return doc_id

    def recall_facts(self, query: str, n_results: int = 5,
                     subject: Optional[str] = None) -> list[dict]:
        """Find facts relevant to a query."""
        where = {"subject": subject} if subject else None
        results = self.facts.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
        )
        return self._unpack(results)

    # ----- Self Notes -----

    def add_self_note(self, note: str, category: str = "general") -> str:
        """Store Annabeth's note about her own behavior."""
        doc_id = f"self_{int(time.time() * 1000)}"
        self.self_notes.add(
            documents=[note],
            metadatas=[{
                "category": category,
                "timestamp": time.time(),
                "type": "self_note",
            }],
            ids=[doc_id],
        )
        return doc_id

    def recall_self_notes(self, query: str, n_results: int = 3) -> list[dict]:
        """Find self-notes relevant to a query."""
        results = self.self_notes.query(
            query_texts=[query],
            n_results=n_results,
        )
        return self._unpack(results)

    # ----- Utility -----

    def recall_all(self, query: str, n_results: int = 3) -> list[dict]:
        """Search across all collections for relevant memories."""
        memories = []
        for coll_name in ("conversations", "facts", "self_notes"):
            coll = getattr(self, coll_name)
            try:
                if coll.count() == 0:
                    continue
                results = coll.query(
                    query_texts=[query],
                    n_results=min(n_results, coll.count()),
                )
                memories.extend(self._unpack(results))
            except Exception as e:
                print(f"[Memory] Skipping {coll_name}: {e}")
                continue
        # Sort by relevance (lower distance = more relevant)
        memories.sort(key=lambda m: m.get("distance", 1.0))
        return memories[:n_results]

    @staticmethod
    def _unpack(results: dict) -> list[dict]:
        """Unpack ChromaDB query results into a flat list of dicts."""
        items = []
        if not results or not results.get("documents"):
            return items
        for i, doc in enumerate(results["documents"][0]):
            item = {"text": doc}
            if results.get("metadatas") and results["metadatas"][0]:
                item.update(results["metadatas"][0][i])
            if results.get("distances") and results["distances"][0]:
                item["distance"] = results["distances"][0][i]
            items.append(item)
        return items
