"""BM25 retrieval with a dependency-free fallback implementation."""
from __future__ import annotations
import math, re
from dataclasses import dataclass
from typing import List
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None
from sqlalchemy.orm import Session
from app.models import Chunk
from app.retrieval.filters import accessible_chunk_query

_TOKEN_RE = re.compile(r"[a-z0-9]+")
def _tokenize(text: str) -> List[str]: return _TOKEN_RE.findall(text.lower())

@dataclass
class BM25Result:
    chunk: Chunk
    score: float

class _SimpleBM25:
    def __init__(self, corpus):
        self.corpus=corpus; self.n=len(corpus); self.k1=1.5; self.b=0.75
        self.avgdl=sum(len(x) for x in corpus)/max(1,self.n)
        self.df={}
        for doc in corpus:
            for t in set(doc): self.df[t]=self.df.get(t,0)+1
    def get_scores(self, query):
        scores=[]
        for doc in self.corpus:
            dl=len(doc); tf={t:doc.count(t) for t in set(doc)}; score=0.0
            for term in query:
                if term not in tf: continue
                idf=math.log(1+(self.n-self.df.get(term,0)+0.5)/(self.df.get(term,0)+0.5))
                f=tf[term]
                score += idf*((f*(self.k1+1))/(f+self.k1*(1-self.b+self.b*dl/max(1,self.avgdl))))
            scores.append(score)
        return scores

def bm25_search(db: Session, query: str, top_k: int = 15, user_id: str | None = None) -> List[BM25Result]:
    chunks = accessible_chunk_query(db, user_id).all()
    if not chunks: return []
    engine = BM25Okapi([_tokenize(c.text) for c in chunks]) if BM25Okapi else _SimpleBM25([_tokenize(c.text) for c in chunks])
    scores = engine.get_scores(_tokenize(query))
    results=[BM25Result(c,float(s)) for c,s in zip(chunks,scores)]
    results.sort(key=lambda x:x.score, reverse=True)
    return results[:top_k]
