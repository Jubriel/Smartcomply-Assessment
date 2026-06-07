import logging
from dotenv import load_dotenv

load_dotenv()


class Config:

    # Logging configuration
    LOG_LEVEL: int = logging.INFO
    LOG_FORMAT: str = ('%(asctime)s - %(funcName)s: %(lineno)d - '
                       '%(levelname)s - %(message)s')
    LOG_DATE_FORMAT: str = '%Y-%m-%d %H:%M:%S'
    LOG_FILE: str = 'src/ass.log'

    # Ollama configuration
    MODEL_NAME: str = "qwen2.5:latest"
    EMBED_MODEL: str = "nomic-text-embed:latest"
    BASE_URL: str = "http://localhost:11434"

    TOP_K: int = 3


SYSTEM_PROMPT = """
You are an adverse media classifier.

Classify a news snippet as:

- adverse: contains allegations, investigations, sanctions,
  regulatory actions, fraud, corruption, criminal activity,
  environmental violations, financial crime, or legal enforcement.

- benign: routine business news, expansion, partnerships,
  product launches, awards, hiring, earnings, or neutral reporting.


Return JSON only:

{
  "snippet_id": "id"
  "adverse": "true" | "false",
  "rationale": "short explanation"
}
"""

config = Config()
