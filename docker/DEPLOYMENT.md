# Dad Bot Production Stack

This stack starts the Dad Bot API with durable backing services out of the box:

- Redis for fast session and task state
- Postgres for durable state and event logs
- PGVector-backed semantic memory when `DADBOT_POSTGRES_DSN` is configured
- Docker volumes for Dad profile, JSON memory, fallback local memory files, and session logs

## Files

- `Dockerfile`: builds the Dad Bot API image
- `docker-compose.yml`: starts `dadbot-api`, `redis`, and `postgres`
- `.env.production.example`: starter environment file for production-like startup

## Quick start

1. Copy `.env.production.example` to `.env.production`.
2. Adjust any model, telemetry, or database values you need.
3. Start the stack:

```powershell
docker compose --env-file .env.production up -d --build
```

4. Check health:

```powershell
docker compose ps
docker compose logs dadbot-api --tail 100
```

The API will be available at `http://localhost:8010/health`.

## Persistent data

The API container writes bot-owned runtime data to mounted volumes:

- `/var/lib/dadbot/dad_profile.json`
- `/var/lib/dadbot/dad_memory.json`
- `/var/lib/dadbot/dad_memory_semantic.sqlite3`
- `/var/log/dadbot/session_logs/`

If `DADBOT_AUTO_INIT_PROFILE=true`, the container will create a starter profile automatically on first boot.

## Environment variables

Core service runtime:

- `DADBOT_API_HOST`
- `DADBOT_API_PORT`
- `DADBOT_API_WORKERS`
- `DADBOT_DEFAULT_MODEL`

Durable backing services:

- `DADBOT_REDIS_URL`
- `DADBOT_POSTGRES_DSN`

PGVector semantic search:

- `DADBOT_SEMANTIC_INDEX_TABLE`
- `DADBOT_SEMANTIC_VECTOR_DIM`
- `DADBOT_SEMANTIC_ANN_INDEX`
- `DADBOT_SEMANTIC_DISTANCE_METRIC`
- `DADBOT_SEMANTIC_HNSW_M`
- `DADBOT_SEMANTIC_HNSW_EF_CONSTRUCTION`
- `DADBOT_SEMANTIC_IVFFLAT_LISTS`

Runtime file paths:

- `DADBOT_PROFILE_PATH`
- `DADBOT_MEMORY_PATH`
- `DADBOT_SEMANTIC_DB_PATH`
- `DADBOT_SESSION_LOG_DIR`
- `DADBOT_AUTO_INIT_PROFILE`

Telemetry:

- `DADBOT_LOG_LEVEL`
- `DADBOT_JSON_LOGS`
- `DADBOT_OTEL_ENABLED`
- `DADBOT_SERVICE_NAME`

## Notes

- This stack does not containerize Ollama. The API still expects Ollama to be reachable from the runtime where the container is deployed.
- For a remote Ollama host, set the appropriate Ollama connectivity environment variables or deploy Dad Bot where Ollama is accessible.
- ANN indexing is only created when both `DADBOT_SEMANTIC_VECTOR_DIM` and `DADBOT_SEMANTIC_ANN_INDEX` are set. Leave them unset if you are still switching embedding models or dimensions.
- For the current embedding candidates, `bge-m3` and `mxbai-embed-large` are typically 1024 dimensions, while `nomic-embed-text` is typically 768. Match this value to the model you actually serve.

## CI

The repository includes a GitHub Actions workflow that runs the live PGVector integration test against a `pgvector/pgvector:pg16` service container on pushes and pull requests.