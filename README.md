## 20 Newsgroups Semantic Search System

This project implements a **semantic search system** over the classic **20 Newsgroups** dataset using:

- **Sentence-transformers** (`all-MiniLM-L6-v2`) for dense text embeddings
- **FAISS** for fast vector similarity search
- **Fuzzy C-means** (via `scikit-fuzzy`) for **fuzzy clustering** of documents
- A custom **semantic cache** keyed by embedding similarity and cluster
- A production-ready **FastAPI** service exposing the search and cache APIs

Run the API with:

```bash
uvicorn main:app --reload
```

The server will be available at `http://localhost:8000`.

---

## 1. Project Structure

Key files and directories:

- **`main.py`**: FastAPI entrypoint.
- **`api/routes.py`**: API endpoints (`/query`, `/cache/stats`, `DELETE /cache`).
- **`core/config.py`**: Central configuration (paths, hyperparameters).
- **`core/data_loader.py`**: 20 Newsgroups download and **text cleaning**.
- **`core/embeddings.py`**: Embedding model + FAISS index build / load.
- **`core/clustering.py`**: **Fuzzy C-means** clustering and silhouette reporting.
- **`core/semantic_cache.py`**: Custom semantic cache (embedding + cluster aware).
- **`core/pipeline.py`**: High-level orchestration tying everything together.
- **`requirements.txt`**: Python dependencies.

Artifact directories:

- **`data/`**: Cached, cleaned 20 Newsgroups documents.
- **`embeddings/`**: Saved embedding matrix, FAISS index, and metadata.
- **`clustering/`**: Fuzzy membership matrix and simple cluster report.

---

## 2. Environment Setup

1. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
.venv\Scripts\activate  # on Windows
# or
source .venv/bin/activate  # on macOS / Linux
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

### Windows note (FAISS)
`faiss-cpu` installation can be environment-sensitive on Windows depending on Python version.
If you hit install errors:

- Use **WSL2** and run the project inside Ubuntu, or
- Use **conda** (`conda install -c conda-forge faiss-cpu`), or
- Run the provided **Dockerfile** (recommended for “it just works” setups).

3. Run the FastAPI app:

```bash
uvicorn main:app --reload
```

4. Open the automatic interactive docs:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## 3. Dataset, Cleaning, and Embeddings

### 3.1 Dataset Download

The project uses **20 Newsgroups** via `sklearn.datasets.fetch_20newsgroups` in `core/data_loader.py`.  
On the first run, it:

- Downloads the full dataset (`subset="all"`).
- Wraps each post in a `Document` dataclass with fields:
  - **`doc_id`**
  - **`text`** (raw)
  - **`cleaned_text`**
  - **`category`** (newsgroup label)

Cleaned documents are cached in `data/20newsgroups_cleaned.jsonl` for fast reloads.

### 3.2 Text Cleaning (Why and How)

Implemented in `_clean_text` in `core/data_loader.py`, we:

- **Remove email headers** like `From:`, `Subject:`, `Lines:` because they are boilerplate and can skew the semantic space toward author metadata, not topic.
- **Strip email addresses** (`user@domain.com`) which are user-specific noise.
- **Heuristically drop signatures**, often separated by `--`, `__`, or similar markers; signatures rarely contribute to topical content.
- **Remove non-alphabetic characters** and normalize to **lowercase** to reduce sparsity and focus the model on content words.
- **Collapse multiple whitespaces** into a single space so the text is clean and embedding-friendly.

Example:

> Before  
> `From: user@xyz.com`  
> `Subject: GPU rendering`  
> `>>> Hello!!!`  
>
> After  
> `gpu rendering hello`

This improves **embedding quality** because the model sees more consistent, topic-focused input.

### 3.3 Embeddings and FAISS Index

In `core/embeddings.py`:

- We load the **`all-MiniLM-L6-v2`** model from `sentence-transformers`.
- Encode all `cleaned_text` into dense vectors.
- **L2-normalize** embeddings and use a **FAISS `IndexFlatIP`**:
  - After normalization, **inner product = cosine similarity**, which is ideal for semantic search.

Persisted artifacts:

- `embeddings/embeddings.npy`: full embedding matrix.
- `embeddings/faiss_index.bin`: FAISS index for fast nearest-neighbor search.
- `embeddings/metadata.json`: document IDs and categories.

On subsequent runs, the system **loads** these artifacts instead of recomputing.

---

## 4. Fuzzy Clustering (Fuzzy C-Means)

Implemented in `core/clustering.py` using **`scikit-fuzzy`** (`skfuzzy.cmeans`).

### 4.1 Why Fuzzy, Not Hard Clustering?

Real topics in 20 Newsgroups overlap:

- A post about **gun laws** can be:
  - **Politics**
  - **Firearms**
  - **Law**

Hard clustering would force it into a single cluster, hiding nuance.  
**Fuzzy C-means** yields **membership scores** for each cluster:

- Example:
  - `Cluster Politics → 0.6`
  - `Cluster Firearms → 0.3`
  - `Cluster Law → 0.1`

This matches how topics actually behave in the dataset.

### 4.2 Model Configuration

- Number of clusters: **`N_CLUSTERS = 20`** (see `core/config.py`).
  - We aim to roughly match the 20 labeled newsgroups, but in a **semantic** space.
- Fuzziness parameter: **`m = 2.0`** (standard, balances hard vs soft).

Embeddings (from `embeddings/embeddings.npy`) are clustered in `core/clustering.py`:

- `u` (membership) has shape `(K, N)` where:
  - `K`: number of clusters
  - `N`: number of documents

Artifacts:

- `clustering/fcm_membership.npy`: membership matrix.
- `clustering/cluster_labels.json`: simple `"Cluster 0"`, `"Cluster 1"`, ...
- `clustering/fcm_membership.txt`: text report including **silhouette score**.

### 4.3 Choosing the Number of Clusters

- We fix `K = 20` for this implementation but justify it via:
  - **Silhouette score** (cosine metric) computed on **hard labels** (argmax of membership).
  - Qualitative inspection of top terms per cluster (not coded exhaustively here but trivial to add).
- A reasonable `K` is one where:
  - Silhouette score is **not too low** (clusters are somewhat separated).
  - Clusters are **interpretable** (e.g., one cluster dominated by `graphics`, `gpu`, `opengl`, `3d` → **Computer Graphics**).

---

## 5. Semantic Cache (Cluster-Aware)

Implemented in `core/semantic_cache.py`.

### 5.1 Why a Semantic Cache?

Users ask the **same question in different words**:

- `What is GPU?`
- `Explain graphics processing units`
- `How do graphics cards work?`

A normal cache would only hit exact strings.  
Our **semantic cache**:

1. Embeds the query.
2. Determines its **dominant cluster** using fuzzy membership (approximated via nearest neighbor).
3. Searches only cached queries **in that cluster**.
4. Computes **cosine similarity**; if it exceeds a threshold → **cache hit**.

This is both **faster** (cluster-restricted lookup) and **smarter** (semantic).

### 5.2 Implementation Details

Cache entries store:

- `query`: original user text.
- `embedding`: L2-normalized embedding (vector).
- `cluster_id`: dominant cluster (int or `None`).
- `result`: list of search results previously computed.

Lookup:

- Iterate only entries with the **same `cluster_id`** (or all, if unknown).
- Track the **best similarity** using dot product (cosine).
- If best similarity ≥ **`similarity_threshold`**, treat as **hit**.

The cache tracks:

- `hit_count`
- `miss_count`
- `hit_rate = hit_count / (hit_count + miss_count)`

### 5.3 Similarity Threshold Trade-offs

Configured in `core/config.py`:

- Default: **`DEFAULT_SIMILARITY_THRESHOLD = 0.85`**

Behavior:

- **Low threshold** (e.g. 0.75):
  - **More cache hits**, but some semantically-distant queries may be incorrectly treated as the same.
  - Example: “How do I install Linux drivers?” might hit a general “Linux kernel overview” result.
- **High threshold** (e.g. 0.92):
  - **Fewer hits**, but the returned cached result is very close in meaning.
  - Many paraphrased queries will still count as misses.

The default **0.85** is a compromise:

- Catches typical paraphrases like “What is GPU?” ↔ “Explain graphics cards”.
- Avoids treating unrelated but loosely related queries as identical.

You can experiment by adjusting `DEFAULT_SIMILARITY_THRESHOLD` and observing:

- `hit_rate` via `/cache/stats`
- qualitative correctness of matched queries (e.g., check `matched_query` and `similarity_score` from `/query` response).

---

## 6. FastAPI API Service

All endpoints are implemented in `api/routes.py` and wired in `main.py`.

### 6.1 Endpoint 1 — `POST /query`

**Request body:**

```json
{
  "query": "How do graphics cards work?"
}
```

**Processing steps:**

1. **Embed query** using the sentence-transformer model.
2. Approximate the **dominant cluster** by:
   - Finding the nearest document in FAISS.
   - Reusing its fuzzy membership to pick the cluster with highest membership.
3. **Semantic cache lookup** restricted to that cluster.
   - If **cache hit**:
     - Return cached result and metadata.
   - If **cache miss**:
     - Run **FAISS search** for top-`k` similar documents.
     - Store the result in the cache under the query’s cluster.

**Response example (shape):**

```json
{
  "query": "How do graphics cards work?",
  "cache_hit": true,
  "matched_query": "What is GPU?",
  "similarity_score": 0.91,
  "result": [
    {
      "document_id": 101,
      "score": 0.87,
      "text_snippet": "gpu rendering performance ...",
      "category": "comp.graphics"
    }
  ],
  "dominant_cluster": 3
}
```

### 6.2 Endpoint 2 — `GET /cache/stats`

Returns cache statistics:

```json
{
  "total_entries": 42,
  "hit_count": 17,
  "miss_count": 25,
  "hit_rate": 0.405
}
```

### 6.3 Endpoint 3 — `DELETE /cache`

Clears the cache and resets counters:

```json
{
  "total_entries": 0,
  "hit_count": 0,
  "miss_count": 0,
  "hit_rate": 0.0
}
```

---

## 7. Running and Testing the System

1. Start the server:

```bash
uvicorn main:app --reload
```

### Optional (recommended): precompute artifacts
The first query can be slow because the system may need to download the dataset and model,
build embeddings, create the FAISS index, and run fuzzy clustering.
You can precompute once:

```bash
python scripts/precompute.py
```

2. Test with `curl` or any HTTP client:

```bash
curl -X POST "http://localhost:8000/query" ^
  -H "Content-Type: application/json" ^
  -d "{\"query\": \"How do graphics cards improve gaming performance?\"}"
```

3. Observe:

- First call: likely **`cache_hit = false`** (miss).
- Rephrase the question:

```bash
curl -X POST "http://localhost:8000/query" ^
  -H "Content-Type: application/json" ^
  -d "{\"query\": \"Explain how GPUs help games run faster\"}"
```

- Now, if similarity ≥ threshold, you should see:
  - **`cache_hit = true`**
  - Non-empty `matched_query`
  - High `similarity_score`

4. Inspect cache stats:

```bash
curl http://localhost:8000/cache/stats
```

---

## 8. Optional: Dockerization (Outline)

If you wish to containerize:

1. Create a `Dockerfile` similar to:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

2. Build and run:

```bash
docker build -t semantic-search-20ng .
docker run -p 8000:8000 semantic-search-20ng
```

---

## 9. Notes and Design Justifications

- **Semantic search** is powered by dense embeddings and cosine similarity (via FAISS), so queries retrieve documents based on **meaning**, not just keyword overlap.
- **Fuzzy clustering** acknowledges that real-world topics overlap; documents can partially belong to multiple semantic themes.
- The **cluster-aware semantic cache** is designed to:
  - Preserve **semantic correctness** (using cosine similarity threshold).
  - Improve **latency** by restricting lookups to a relevant cluster.
- All heavy artifacts (dataset, embeddings, clustering) are **cached on disk**, so the system behaves realistically: first run is expensive, subsequent runs are fast.

