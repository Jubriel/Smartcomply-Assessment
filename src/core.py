# type: ignore
from typing import List, Dict, Any

import chromadb
from pydantic_ai import Embedder, Agent
from sentence_transformers import CrossEncoder

from src.config import config, SYSTEM_PROMPT
from src.data import sanctioned_entities
from src.schema import ClassificationOutput, FinalOutput, RetrievalOutput
from src.utils import setup_logging

logger = setup_logging()


class Screening:
    def __init__(self):
        self.agent = Agent(config.MODEL_NAME,
                           instructions=SYSTEM_PROMPT,
                           output_type=ClassificationOutput)

        self.embed = Embedder(config.EMBED_MODEL)
        self.client = chromadb.Client()
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.collection = self.client.get_or_create_collection(
            name="sanctioned_entities",
            embedding_function=self.embed
        )
        self.collection.add(
            documents=sanctioned_entities,
            ids=[str(i) for i in range(len(sanctioned_entities))]
        )

    async def retriever(self, query: str, top_k: int = config.TOP_K
                        ) -> List[Dict[str, Any]]:
        """
        Retrieve and rerank the most relevant entities/documents.
        """
        try:
            query_embedding = await self.embed.embed_query(query)

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k * 3,  # retrieve extra candidates
                include=["documents", "distances"]
            )

            documents = results.get("documents", [[]])[0]
            distances = results.get("distances", [[]])[0]

            if not documents:
                return []

            # Convert distance -> similarity
            candidates = []
            for doc, distance in zip(documents, distances):
                candidates.append({
                    "matched_name": doc,
                    "score": round(1 - distance, 4)
                })

            # Cross-encoder reranking
            rerank_pairs = [(query, item["matched_name"])
                            for item in candidates]
            rerank_scores = self.reranker.predict(rerank_pairs)

            for item, score in zip(candidates, rerank_scores):
                item["rerank_score"] = float(score)

            # Sort by reranker score
            ranked = sorted(
                candidates,
                key=lambda x: x["rerank_score"],
                reverse=True
            )

            return ranked[:top_k]
        except Exception as e:
            logger.exception("Error retrieving entities", exc_info=e)
            raise

    async def classify(self, snip: dict) -> ClassificationOutput:
        """
        Classify if the query matches any sanctioned entity.
        """
        try:
            query = snip.snippet
            result = await self.agent.run(query=query)
            return ClassificationOutput(
                snippet_id=snip.snippet_id,
                adverse=result.adverse,
                rationale=result.rationale
            )
        except Exception as e:
            logger.exception("Error classifying entity", exc_info=e)
            raise

    async def process_snippets(self, query: str) -> FinalOutput:
        """
        Process a list of snippets and classify them.
        """
        try:
            ranked_entities = await self.retriever(query)
            logger.info(f"Top retrieved entities: {ranked_entities}")

            classifications = []
            for snip in ranked_entities:
                classification = await self.classify(snip)
                classifications.append(classification)
            return FinalOutput(
                watchlist=[RetrievalOutput(**item)
                           for item in ranked_entities],
                classification=classifications
            )
        except Exception as e:
            logger.exception("Error processing snippets", exc_info=e)
            raise