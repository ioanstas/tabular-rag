"""Search saved summary vectors by embedding only the incoming question."""
from table_rag.dense_index import DenseIndex, embed_texts, validate_dense_index
from table_rag.sparse_retrieval import RetrievalResult


def search_dense(index: DenseIndex, model, question: str, top_k: int = 5) -> list[RetrievalResult]:
    """Return cosine-ranked tables using a model from load_embedding_model().

    Reuse the loaded index and pinned model across questions in an application.
    Scores measure similarity, not answer confidence. Ties sort by table ID.
    """
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be non-empty text")
    validate_dense_index(index)
    query_vector = embed_texts(model, [question])[0]
    # Unit-length vectors make dot products equal cosine similarities.
    scores = index.vectors @ query_vector
    positions = sorted(
        range(len(index.table_ids)),
        key=lambda position: (-float(scores[position]), index.table_ids[position]),
    )[:top_k]
    return [
        RetrievalResult(index.table_ids[position], rank, float(scores[position]))
        for rank, position in enumerate(positions, start=1)
    ]
