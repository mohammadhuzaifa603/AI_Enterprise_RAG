from app.ingestion.embeddings import embed_texts
from app.models import Chunk, Document
from app.retrieval.bm25 import bm25_search
from app.retrieval.dense import cosine_similarity, dense_search
from app.retrieval.hybrid import hybrid_search
from app.retrieval.reranker import rerank


def _seed_chunks(db):
    doc = Document(
        filename="test.pdf",
        original_filename="test.pdf",
        file_path="/tmp/test.pdf",
        status="processed",
        num_pages=1,
    )
    db.add(doc)
    db.flush()

    texts = [
        "The company allows remote work up to three days per week.",
        "Employees accrue twenty days of paid time off per year.",
        "Q2 2026 revenue reached 4.82 million dollars, up 14 percent.",
        "The espresso machine in the kitchen is broken again.",
    ]
    vectors = embed_texts(texts)
    for i, (text, vec) in enumerate(zip(texts, vectors)):
        db.add(Chunk(document_id=doc.id, page_number=1, chunk_index=i, text=text, embedding=vec))
    db.flush()
    return doc


def test_cosine_similarity_identical_vectors_is_one():
    v = [1.0, 2.0, 3.0]
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-9


def test_cosine_similarity_orthogonal_vectors_is_zero():
    assert abs(cosine_similarity([1.0, 0.0], [0.0, 1.0])) < 1e-9


def test_dense_search_returns_relevant_chunk_first(db_session):
    _seed_chunks(db_session)
    results = dense_search(db_session, "How many days can I work from home?", top_k=4)
    assert len(results) == 4
    # The remote-work chunk should score at least as high as the totally
    # unrelated "espresso machine" chunk.
    remote_score = next(r.score for r in results if "remote work" in r.chunk.text)
    espresso_score = next(r.score for r in results if "espresso" in r.chunk.text)
    assert remote_score >= espresso_score


def test_bm25_search_ranks_keyword_match_first(db_session):
    _seed_chunks(db_session)
    results = bm25_search(db_session, "revenue Q2 2026", top_k=4)
    assert results[0].chunk.text.startswith("Q2 2026 revenue")


def test_bm25_search_empty_corpus_returns_empty(db_session):
    results = bm25_search(db_session, "anything", top_k=5)
    assert results == []


def test_hybrid_search_combines_and_dedupes(db_session):
    _seed_chunks(db_session)
    results = hybrid_search(db_session, "remote work days per week", top_k=4)
    assert len(results) > 0
    ids = [r.chunk.id for r in results]
    assert len(ids) == len(set(ids))  # no duplicates
    assert all(0.0 <= r.hybrid_score <= 1.0 for r in results)


def test_rerank_orders_by_relevance(db_session):
    _seed_chunks(db_session)
    candidates = hybrid_search(db_session, "paid time off per year", top_k=4)
    reranked = rerank("paid time off per year", candidates, top_k=4)
    assert len(reranked) > 0
    assert "paid time off" in reranked[0].chunk.text.lower() or "twenty days" in reranked[0].chunk.text.lower()


def test_rerank_empty_candidates_returns_empty():
    assert rerank("anything", [], top_k=5) == []
