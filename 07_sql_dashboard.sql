-- Databricks notebook source

-- MAGIC %md
-- MAGIC # 07 — SQL Observability Dashboard
-- MAGIC
-- MAGIC **Purpose:** Monitor the RAG endpoint using the inference table that was
-- MAGIC enabled in `05_model_serving.py`. Every request and response is automatically
-- MAGIC logged to a Delta table in Unity Catalog.
-- MAGIC
-- MAGIC **Inference table:** `main.genai_demo.genai_demo_rag_endpoint_payload`
-- MAGIC
-- MAGIC **What this dashboard shows:**
-- MAGIC 1. Overview — total requests, success rate, avg latency
-- MAGIC 2. Daily request trend
-- MAGIC 3. Latency distribution (p50 / p95 / p99)
-- MAGIC 4. Recent Q&A pairs (question + answer extracted from JSON)
-- MAGIC 5. Top source documents cited in answers
-- MAGIC 6. Error analysis — failed requests
-- MAGIC
-- MAGIC **Run after:** `05_model_serving.py` with inference tables enabled

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 0. Verify Inference Table Exists

-- COMMAND ----------

SHOW TABLES IN main.genai_demo LIKE '*payload*';

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 1. Overview — Key Metrics

-- COMMAND ----------

SELECT
  COUNT(*)                                                  AS total_requests,
  ROUND(AVG(execution_time_ms), 0)                         AS avg_latency_ms,
  ROUND(PERCENTILE(execution_time_ms, 0.50), 0)            AS p50_latency_ms,
  ROUND(PERCENTILE(execution_time_ms, 0.95), 0)            AS p95_latency_ms,
  SUM(CASE WHEN status_code = 200 THEN 1 ELSE 0 END)       AS successful_requests,
  SUM(CASE WHEN status_code != 200 THEN 1 ELSE 0 END)      AS failed_requests,
  ROUND(
    100.0 * SUM(CASE WHEN status_code = 200 THEN 1 ELSE 0 END) / COUNT(*), 1
  )                                                         AS success_rate_pct,
  MIN(DATE(timestamp))                                      AS first_request_date,
  MAX(DATE(timestamp))                                      AS last_request_date
FROM main.genai_demo.genai_demo_rag_endpoint_payload;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 2. Daily Request Trend

-- COMMAND ----------

SELECT
  DATE(timestamp)                                          AS request_date,
  COUNT(*)                                                 AS total_requests,
  SUM(CASE WHEN status_code = 200 THEN 1 ELSE 0 END)      AS successful,
  SUM(CASE WHEN status_code != 200 THEN 1 ELSE 0 END)     AS failed,
  ROUND(AVG(execution_time_ms), 0)                        AS avg_latency_ms
FROM main.genai_demo.genai_demo_rag_endpoint_payload
GROUP BY DATE(timestamp)
ORDER BY request_date DESC;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 3. Latency Distribution (Hourly)

-- COMMAND ----------

SELECT
  DATE_TRUNC('hour', timestamp)                            AS hour,
  COUNT(*)                                                 AS requests,
  ROUND(PERCENTILE(execution_time_ms, 0.50), 0)           AS p50_ms,
  ROUND(PERCENTILE(execution_time_ms, 0.95), 0)           AS p95_ms,
  ROUND(PERCENTILE(execution_time_ms, 0.99), 0)           AS p99_ms,
  ROUND(MAX(execution_time_ms), 0)                        AS max_ms
FROM main.genai_demo.genai_demo_rag_endpoint_payload
WHERE status_code = 200
GROUP BY DATE_TRUNC('hour', timestamp)
ORDER BY hour DESC
LIMIT 48;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 4. Recent Q&A Pairs
-- MAGIC
-- MAGIC Extracts the question and answer from the JSON request/response payloads.
-- MAGIC
-- MAGIC Request format:  `{"dataframe_records": [{"question": "..."}]}`
-- MAGIC Response format: `{"predictions": [{"answer": "...", "sources": "..."}]}`

-- COMMAND ----------

SELECT
  timestamp,
  execution_time_ms                                                              AS latency_ms,
  status_code,
  -- Extract question from request JSON
  GET_JSON_OBJECT(request,  '$.dataframe_records[0].question')                  AS question,
  -- Extract answer from response JSON
  GET_JSON_OBJECT(response, '$.predictions[0].answer')                          AS answer
FROM main.genai_demo.genai_demo_rag_endpoint_payload
WHERE status_code = 200
ORDER BY timestamp DESC
LIMIT 25;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 5. Top Source Documents Cited
-- MAGIC
-- MAGIC Parses the `sources` JSON array from each response to count how often
-- MAGIC each source document was retrieved and cited in an answer.

-- COMMAND ----------

WITH sources_exploded AS (
  SELECT
    timestamp,
    EXPLODE(
      FROM_JSON(
        GET_JSON_OBJECT(response, '$.predictions[0].sources'),
        'ARRAY<STRUCT<filename STRING, chunk_text STRING, score DOUBLE>>'
      )
    ) AS source
  FROM main.genai_demo.genai_demo_rag_endpoint_payload
  WHERE status_code = 200
    AND GET_JSON_OBJECT(response, '$.predictions[0].sources') IS NOT NULL
)
SELECT
  source.filename                        AS source_document,
  COUNT(*)                               AS times_cited,
  ROUND(AVG(source.score), 4)            AS avg_relevance_score,
  ROUND(MIN(source.score), 4)            AS min_score,
  ROUND(MAX(source.score), 4)            AS max_score
FROM sources_exploded
GROUP BY source.filename
ORDER BY times_cited DESC;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 6. Question Frequency — Most Common Questions

-- COMMAND ----------

SELECT
  GET_JSON_OBJECT(request, '$.dataframe_records[0].question')  AS question,
  COUNT(*)                                                      AS times_asked,
  ROUND(AVG(execution_time_ms), 0)                             AS avg_latency_ms,
  MAX(DATE(timestamp))                                         AS last_asked
FROM main.genai_demo.genai_demo_rag_endpoint_payload
WHERE status_code = 200
GROUP BY GET_JSON_OBJECT(request, '$.dataframe_records[0].question')
ORDER BY times_asked DESC
LIMIT 20;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 7. Error Analysis — Failed Requests

-- COMMAND ----------

SELECT
  status_code,
  COUNT(*)                                                     AS error_count,
  MIN(timestamp)                                               AS first_seen,
  MAX(timestamp)                                               AS last_seen,
  -- Show the raw response for the most recent error of each type
  FIRST_VALUE(response) OVER (
    PARTITION BY status_code ORDER BY timestamp DESC
  )                                                            AS sample_error_response
FROM main.genai_demo.genai_demo_rag_endpoint_payload
WHERE status_code != 200
GROUP BY status_code, response, timestamp
ORDER BY error_count DESC;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 8. Latency Buckets — Distribution Histogram

-- COMMAND ----------

SELECT
  CASE
    WHEN execution_time_ms <  500  THEN '< 0.5s'
    WHEN execution_time_ms <  1000 THEN '0.5s – 1s'
    WHEN execution_time_ms <  2000 THEN '1s – 2s'
    WHEN execution_time_ms <  5000 THEN '2s – 5s'
    WHEN execution_time_ms < 10000 THEN '5s – 10s'
    ELSE '> 10s'
  END                                   AS latency_bucket,
  COUNT(*)                              AS request_count,
  ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct_of_total
FROM main.genai_demo.genai_demo_rag_endpoint_payload
WHERE status_code = 200
GROUP BY 1
ORDER BY MIN(execution_time_ms);

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## ✓ Dashboard Complete
-- MAGIC
-- MAGIC Pin any of these query results to a **Databricks SQL Dashboard** for a live view:
-- MAGIC 1. Open **Databricks SQL → Dashboards → New Dashboard**
-- MAGIC 2. Add each query above as a widget (table, bar chart, counter)
-- MAGIC 3. Set auto-refresh to keep metrics up to date
