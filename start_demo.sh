#!/bin/bash
cd ~/github/semdriftbd
source venv/bin/activate

echo "=== SemDriftBD Demo Startup ==="

# Step 1: Start Docker cluster
echo "[1/5] Starting Spark cluster + Kafka..."
docker-compose up -d
docker-compose -f kafka-docker-compose.yml up -d
sleep 15

# Step 2: Verify cluster
echo "[2/5] Verifying cluster..."
curl -s http://localhost:8090/json/ | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(f'  Spark Master: {d[\"status\"]}')
print(f'  Workers alive: {d[\"aliveworkers\"]}')
print(f'  Total cores: {d[\"cores\"]}')
print(f'  Total memory: {d[\"memory\"]} MB')
"

# Step 3: Run pipeline to generate fresh results
echo "[3/5] Running pipeline..."
python3 src/compute_all_drift.py
python3 src/adaptive_threshold.py

# Step 4: Stream articles to Kafka
echo "[4/5] Streaming articles to Kafka..."
python3 src/kafka_producer.py &
PRODUCER_PID=$!
sleep 5

# Step 5: Start dashboards
echo "[5/5] Starting dashboards..."
streamlit run src/dashboard.py --server.port 8501 --server.fileWatcherType none &
sleep 3
streamlit run src/demo.py --server.port 8502 --server.fileWatcherType none &
sleep 3

echo ""
echo "=== ALL SYSTEMS READY ==="
echo ""
echo "BROWSER TABS TO OPEN:"
echo "  1. http://localhost:8501  — SemDriftBD Dashboard (main demo)"
echo "  2. http://localhost:8502  — Full Demo Overview"
echo "  3. http://localhost:8090  — Spark Cluster (3 workers)"
echo "  4. http://localhost:4040  — Spark Job UI"
echo ""
echo "DEMO ORDER:"
echo "  1. Show 8501 — WKCS chart, peak drift, attribution"
echo "  2. Show 8090 — 3 worker nodes alive"
echo "  3. Run streaming: python3 src/spark_structured_streaming.py"
echo "     Then open: http://localhost:4040/StreamingQuery"
echo "  4. Show 8502 — scalability charts, Delta Lake"
echo ""
echo "Press Ctrl+C to stop all services"

# Keep running
wait $PRODUCER_PID
