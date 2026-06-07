# Section 1: ML Fundamentals

## 1a. Bias-variance and Regularisation

### Diagnosis

The model achieving 99.2% accuracy on the training set but dropping to 74% on the test set is a classic case of high variance (overfitting). The model has memorized the training data noise rather than learning generalizable patterns.  

### Interventions

- Reduce Tree Depth / Complexity: Limits how granular the decision boundaries can become. Trade-off: Too shallow, and the model introduces high bias (underfitting), missing complex fraud patterns.
- Increase L1/L2 Regularisation: Penalizes large leaf weights, forcing smoother predictions. Trade-off: Can reduce the impact of genuinely strong predictive features if tuned too aggressively.
- Increase Subsampling (Rows/Columns): Training each tree on a fraction of data/features. Trade-off: Increases training time slightly and might require more trees to reach convergence.

Class Imbalance (1% fraud rate): Accuracy is a highly misleading metric here; predicting "not fraud" every time yields 99% accuracy but fails the business objective. You should prioritize PR-AUC (Precision-Recall Area Under Curve) and the F1-Score. In compliance, Recall is critical (catching as much fraud as possible), but Precision must be balanced so compliance teams aren't overwhelmed with false positives.

## 1b. Embeddings and Similarity Search

- Dense vs. Sparse: Dense embeddings map text to fixed-dimensional continuous vectors, capturing semantic meaning (e.g., "bank" and "financial institution" are close). Sparse representations (like TF-IDF) map text to high-dimensional, mostly zero vectors based on exact word frequencies, lacking contextual understanding.  
- Watchlist Matching Architecture: A Hybrid approach is best. BM25 is excellent for exact keyword matches and partial string overlaps (useful for minor spelling errors), while a vector database captures semantic similarity and complex transliterations that don't share exact characters.
- Model Selection &Evaluation: A pre-trained sentence transformer (e.g., all-MiniLM-L6-v2) is a good baseline. Evaluate using Mean Reciprocal Rank (MRR) or Normalized Discounted Cumulative Gain (NDCG) against a curated, human-labeled dataset of known transliterations and typos specific to your region.

## 1c. Anomaly Detection

- Algorithms:

  - Isolation Forest: Efficient for high-dimensional data, it isolates anomalies by randomly partitioning feature thresholds. It assumes anomalies are "few and different."

  - Autoencoders: A neural network that learns to compress and reconstruct normal data. Anomalies are flagged when the reconstruction error is exceptionally high.

- Concept Drift: Implement a rolling-window retraining strategy (e.g., retraining monthly on the last 90 days) and monitor feature distribution shifts over time to trigger ad-hoc retraining.

- Evaluation without Ground Truth: Rely on proxy metrics like the "alert rate stability" (the percentage of flagged transactions should remain relatively constant). Implement a human-in-the-loop system where compliance officers sample and label a small, random batch of alerts to estimate precision.  

## 1d. Feature Engineering for Compliance

- Features:
  - `time_since_last_txn`: Detects sudden bursts in velocity.
  - `amount_zscore_7d`: Flags amounts deviating from the entity's weekly norm.
  - `is_high_risk_country_dest`: Binary flag based on FATF watchlists.
  - `channel_change_flag`: Indicates if the transaction medium (e.g., mobile to web) changed abruptly.
  - `weekend_txn_ratio`: Proportion of off-hours activity.
  - `sender_receiver_geo_distance`: Physical distance between entities.
  - `daily_txn_count`: Simple velocity metric.
  - `avg_amount_rolling_30d`: Baseline moving average for the sender.

- Missing Values & High-Cardinality: Impute numericals with the median (robust to outliers). For categorical columns like country codes, replace missing values with a distinct `UNKNOWN` category, and use Target Encoding to handle the high cardinality.
- Data Leakage: The biggest risk is using future information (e.g., a 30-day average that includes data after the transaction timestamp). Prevent this by ensuring strict chronological splitting during train/test splits and using window functions that only look backward.


# Section 3: MLOps & Production Systems

## 3a. Model Monitoring Design

- Data Drift: Track Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) tests on key features like transaction amounts, calculated daily against a baseline training distribution.
- Performance Degradation: With a 30-day label lag, rely heavily on monitoring prediction distributions (e.g., sudden spikes in the average assigned risk score) and operational proxy metrics like customer support ticket volume regarding frozen funds.
- Alerting: A slight drift in input features triggers an automated retrain. A sudden drop in throughput or a massive spike in extreme risk scores triggers an immediate rollback to the previous model version.  

## 3b. Scaling a Screening Service

- Embedding Inference: CPU batching is often sufficient for lightweight sentence transformers and cheaper, but GPU deployment is necessary for high-throughput LLM architectures.
- Vector Latency: Keep p99 under 200ms by utilizing HNSW (Hierarchical Navigable Small World) indexes within a dedicated vector database (like Qdrant) rather than flat exhaustive search.
- Queues: Introduce Redis Streams/Celery for the heavy RAG/LLM synthesis tasks. The initial vector search/screening should remain synchronous on the API gateway to fulfill real-time block/allow logic.  

## 3c. CI/CD

- Versioning: Use DVC (Data Version Control) to pin dataset hashes and model artifacts to specific Git commits of your training code.
- Pipeline Stages:  
  - Build: Linting and Unit tests.
  - Staging: Integration tests (checking DB connections) and Model quality checks (evaluating F1-score against a golden dataset).
  - Production: Canary deployment. Rollback if the canary shows a 5% increase in 500 errors or p90 latency exceeds 500ms.


# Section 4: End-to-End System Design Case Study

## Data Flow & Component Breakdown:

- Ingestion: Raw transactions hit a Kafka topic.Synchronous Layer (< 3s) : FastAPI workers consume the event, calling the Vector DB for fuzzy watchlist matching. If clear, the transaction proceeds.
- Asynchronous Layer (Behavioural & RAG): Parallel workers analyze the 90-day rolling window for anomalies. If flagged, an async task triggers the RAG pipeline to synthesize adverse media.
Dashboarding: Results, features, and the LLM rationale  are indexed into Elasticsearch/PostgreSQL for the Compliance Officer UI.

## Trade-offs:  

- Synchronous vs. Asynchronous: I placed behavioral analytics in the async layer. Why: Calculating complex 90-day aggregations in real-time jeopardizes the 3-second SLA. It's better to process the transaction and flag the account post-facto than drop the transaction entirely due to a timeout.

Handling False Positives: Implement a feedback loop in the compliance UI where officers explicitly mark "True Positive" or "False Positive". This data is fed back into a secondary classifier that learns the patterns of false positives, applying a dampening weight to future similar alerts.