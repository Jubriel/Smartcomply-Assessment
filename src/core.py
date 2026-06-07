# type: ignore
import os
from typing import List, Dict, Any

import chromadb
from pydantic_ai import Embedder, Agent
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider
from sentence_transformers import CrossEncoder

from src.config import config, SYSTEM_PROMPT
from src.data import sanctioned_entities, news_snippets
from src.schema import ClassificationOutput, FinalOutput, RetrievalOutput
from src.utils import setup_logging

logger = setup_logging()

formatted_base_url = config.BASE_URL.rstrip('/')
if formatted_base_url.endswith('/api'):
    formatted_base_url = formatted_base_url.replace('/api', '/v1')
elif not formatted_base_url.endswith('/v1'):
    formatted_base_url = f"{formatted_base_url}/v1"


model = OllamaModel(
    model_name=config.MODEL_NAME,
    provider=OllamaProvider(
        base_url=formatted_base_url
    )
)

embed = OllamaModel(
    model_name=config.EMBED_MODEL,
    provider=OllamaProvider(
        base_url=config.BASE_URL
    )
)


class Screening:
    def __init__(self):
        self.agent = Agent(model=model,
                           instructions=SYSTEM_PROMPT,
                           output_type=ClassificationOutput)
        # 1. Format the URL for OpenAI compatibility (required for Ollama embeddings)
        base_url = config.BASE_URL.rstrip('/')
        if base_url.endswith('/api'):
            embed_url = base_url.replace('/api', '/v1')
        elif not base_url.endswith('/v1'):
            embed_url = f"{base_url}/v1"
        else:
            embed_url = base_url

        # 2. Tell the underlying client where to look
        os.environ['OPENAI_BASE_URL'] = embed_url
        os.environ['OPENAI_API_KEY'] = 'ollama'

        # 3. Use the 'openai:' prefix to force the compatibility route
        self.embed = Embedder("openai:nomic-embed-text")

        # self.embed = Embedder(model="ollama:nomic-embed-text")
        self.client = chromadb.Client()
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.collection = self.client.get_or_create_collection(
            name="sanctioned_entities",
        )

    @classmethod
    async def create(cls):
        self = cls()
        await self.initialize()
        return self

    async def initialize(self):
        try:
            count = self.collection.count()

            if count > 0:
                return

            emb_result = await self.embed.embed(inputs=sanctioned_entities,
                                                input_type="query")

            self.collection.upsert(
                ids=[str(i) for i in range(len(sanctioned_entities))],
                documents=sanctioned_entities,
                embeddings=emb_result.embeddings
            )
        except Exception as e:
            logger.exception("Error initializing collection", exc_info=e)
            raise

    async def retriever(self, query: str, top_k: int = config.TOP_K
                        ) -> List[Dict[str, Any]]:
        """
        Retrieve and rerank the most relevant entities/documents.
        """
        try:
            emb_result = await self.embed.embed(inputs=query,
                                                input_type="query")
            query_embedding = emb_result.embeddings[0]

            results = self.collection.query(
                query_embeddings=[query_embedding],
                # Pass the raw list of floats
                n_results=top_k * 3,
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
            news = snip['snippet']
            response = await self.agent.run(news)
            result = response.output
            return ClassificationOutput(
                snippet_id=snip["snippet_id"],
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
            # 1. Retrieve the sanctioned entities
            ranked_entities = await self.retriever(query)
            logger.info(f"Top retrieved entities: {ranked_entities}")

            classifications = []

            # 2. Loop through the retrieved entities
            for ent in ranked_entities:
                # 3. Find all snippets in the global list
                # that match this specific entity
                matching_snippets = [
                    snip for snip in news_snippets
                    if snip["entity"] == ent["matched_name"]
                ]

                # 4. Run the classification agent on the actual snippet data,
                # not the vector result
                for snip in matching_snippets:
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