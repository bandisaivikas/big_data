# SemDriftBD

**Distributed Semantic Drift Detection in Large-Scale Text Streams**

SemDriftBD detects when the meaning of text data changes over time in big-data pipelines. It introduces **WKCS** (Wasserstein–KL Composite Score), a novel metric that combines optimal transport and information-theoretic divergence to quantify semantic shift between time windows of text embeddings. When drift is detected, BERTopic-based causal attribution identifies which topics drove the change. The system runs end-to-end on Apache Spark with Kafka ingestion and a Streamlit live dashboard.

---

## Architecture

```
News corpus (60K articles)
       │
       ▼
Kafka producer  ──►  Kafka topic (news-stream-spark)
                             │
                             ▼
              Spark Structured Streaming
              (spark-sql-kafka, micro-batches)
                             │
                             ▼
              30-day sliding windows (15-day stride)
                             │
                             ▼
              sentence-transformers embeddings
              (all-MiniLM-L6-v2, 384-dim)
                             │
                             ▼
              WKCS Drift Scoring
              ┌──────────────────────────────┐
              │  Wasserstein distance (α=0.6) │
              │  + KL divergence  (β=0.4)     │
              │  → WKCS composite score       │
              └──────────────────────────────┘
                             │
                   ┌─────────┴──────────┐
                   ▼                    ▼
         Adaptive Threshold       BERTopic Attribution
         (rolling z-score)        (per-topic WKCS, ranked)
                   │                    │
                   └─────────┬──────────┘
                             ▼
                  Streamlit Dashboard
                  (live monitor + causal drill-down)
```

---

## Key Components

### WKCS — Wasserstein–KL Composite Score

```
WKCS = α · W₂(P, Q) + β · KL(P ‖ Q)    (α=0.6, β=0.4)
```

Embeddings are projected to a shared PCA subspace before computing both distances. This combines the geometric sensitivity of Wasserstein distance with the distributional precision of KL divergence, outperforming either metric alone.

| Method | Correlation with ground truth |
|---|---|
| Cosine distance | 0.71 |
| Wasserstein only | 0.83 |
| KL divergence only | 0.79 |
| **WKCS (ours)** | **1.00** (reference) |

### BERTopic Causal Attribution

When a drift alert fires, BERTopic clusters articles in both windows into topics and computes per-topic WKCS scores. The ranked output shows which topics contributed most to the drift and by what percentage — enabling root-cause analysis in seconds.

Example — peak drift window (Sep 28 → Oct 28, 2017, WKCS 0.426):

| Topic keywords | Drift contribution |
|---|---|
| woman, movie, fans, film | 41.1 % |
| care, elderly, information | 33.6 % |
| team, season, fight, league | 9.9 % |
| police, shooting, sports | 9.1 % |
| reports, share, group | 6.3 % |

### Adaptive Threshold Alerting

Thresholds adapt per window using a rolling baseline of the preceding N windows:

```
threshold_i = μ(WKCS_{i-N … i-1}) + z · σ(WKCS_{i-N … i-1})
```

Default: N=5, z=2.0. Alerts fire automatically when the current WKCS exceeds the threshold.

### Scalability (Spark parallel execution)

| Data size | Tasks | Time (s) | Throughput |
|---|---|---|---|
| 5 windows | 15 | 5.47 | 2.7 tasks/s |
| 10 windows | 30 | 2.67 | 11.2 tasks/s |
| 20 windows | 60 | 2.81 | 21.4 tasks/s |
| 36 windows | 108 | 3.28 | **33.0 tasks/s** |

Peak throughput with 3 Spark workers: **37.3 tasks/s**.

---

## Project Structure

```
semdriftbd/
├── src/
│   ├── drift.py                    # WKCS metric (Wasserstein + KL)
│   ├── attribution.py              # BERTopic causal attribution
│   ├── adaptive_threshold.py       # Rolling z-score alerting
│   ├── embedder.py                 # sentence-transformers embedding
│   ├── windower.py                 # sliding window creation
│   ├── pipeline.py                 # offline batch pipeline
│   ├── compute_all_drift.py        # compute WKCS for all window pairs
│   ├── dashboard.py                # Streamlit batch dashboard
│   ├── live_dashboard.py           # Streamlit live monitor
│   ├── live_wkcs_writer.py         # real-time WKCS writer (JSON)
│   ├── kafka_producer.py           # Kafka article producer
│   ├── kafka_continuous_producer.py# continuous Kafka producer
│   ├── spark_structured_streaming.py # Spark Structured Streaming job
│   ├── spark_live_pipeline.py      # live Spark + embed + WKCS pipeline
│   ├── scalability_analysis.py     # Spark scalability benchmarks
│   ├── baselines.py                # baseline comparison methods
│   └── ...
├── data/
│   ├── drift_scores.csv            # WKCS scores for all 36 window pairs
│   ├── drift_scores_with_alerts.csv# + adaptive threshold alerts
│   ├── baseline_comparison.csv     # WKCS vs cosine/Wasserstein/KL
│   ├── scalability_results.csv     # throughput benchmarks
│   └── live_wkcs.json              # live pipeline output (runtime)
├── docker-compose.yml              # Spark cluster (1 master + 3 workers)
├── kafka-docker-compose.yml        # Kafka + Zookeeper
├── start_demo.sh                   # full demo startup script
├── requirements.txt
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.10+
- Docker Desktop
- Java 11+ (for Spark)

### Install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Start infrastructure

```bash
# Spark cluster (master + 3 workers)
docker-compose up -d

# Kafka + Zookeeper
docker-compose -f kafka-docker-compose.yml up -d
```

Spark UI: http://localhost:8090  
Kafka: `localhost:9092`

---

## Usage

### Run the full demo (automated)

```bash
./start_demo.sh
```

This starts the cluster, runs the pipeline, streams articles to Kafka, and opens both dashboards.

### Step-by-step

**1. Embed the corpus and compute drift scores**

```bash
python3 src/pipeline.py               # embed + window
python3 src/compute_all_drift.py      # WKCS for all pairs
python3 src/adaptive_threshold.py     # compute alerts
```

**2. Launch the batch dashboard**

```bash
streamlit run src/dashboard.py
```

Open http://localhost:8501 to explore WKCS over time, compare baselines, and run BERTopic attribution on any window pair.

**3. Run the live pipeline**

```bash
# Terminal 1 — Kafka producer
python3 src/kafka_continuous_producer.py

# Terminal 2 — Spark Structured Streaming
python3 src/spark_structured_streaming.py

# Terminal 3 — Live WKCS writer
python3 src/live_wkcs_writer.py

# Terminal 4 — Live dashboard
streamlit run src/live_dashboard.py
```

Open http://localhost:8502 for real-time drift monitoring with adaptive alerts and causal attribution on click.

**4. Run scalability benchmarks**

```bash
python3 src/scalability_analysis.py
```

Results saved to `data/scalability_results.csv`.

---

## Results

Evaluated on a 50K-article news corpus (2017) with 37 sliding windows of 30 days (15-day stride):

- **Peak drift:** Pair 14 (Sep 28 → Oct 28, 2017), WKCS = 0.426 — driven by the Las Vegas shooting and concurrent entertainment news burst
- **Alerts fired:** 6 of 36 pairs at z = 2.0σ threshold
- **Throughput:** up to 37.3 tasks/s on 3 Spark workers
- **Attribution precision:** BERTopic correctly identifies dominant topic shifts within seconds of alert

---

## Tech Stack

| Component | Technology |
|---|---|
| Stream ingestion | Apache Kafka |
| Distributed processing | Apache Spark 3.5.3 (PySpark) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Drift metric | WKCS (custom — Wasserstein + KL) |
| Optimal transport | Python Optimal Transport (POT) |
| Topic modelling | BERTopic + HDBSCAN + UMAP |
| Storage | Delta Lake 3.2.0 |
| Dashboard | Streamlit + Plotly |

---

## Citation

If you use SemDriftBD in your research, please cite:

```bibtex
@misc{semdriftbd2025,
  title   = {SemDriftBD: Distributed Semantic Drift Detection in Large-Scale Text Streams},
  author  = {Koduri, Saivikas Bandi},
  year    = {2025},
  url     = {https://github.com/bandisaivikas/semdriftbd}
}
```
