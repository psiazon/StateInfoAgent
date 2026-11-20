# StateInfoAgent conversational memory

## Local Postman test

Set `MEMORY_BACKEND=memory` (default) and start the FastAPI app as before.

POST `/api/agent/chat`:
```json
{"message":"What is the population of Illinois?","thread_id":"demo-001"}
```
Then POST:
```json
{"message":"What is its capital?","thread_id":"demo-001"}
```
Use the same `thread_id` to continue; omit it to start a new conversation (the response returns the generated ID). Use a different ID for a separate conversation.

## Azure durable memory

Set App Service environment variables:
- `MEMORY_BACKEND=postgres`
- `POSTGRES_CONNECTION_STRING=postgresql://USER:PASSWORD@HOST:5432/DB?sslmode=require`

Provision an Azure Database for PostgreSQL Flexible Server and grant the app appropriate network/database access. Prefer managed identity/Entra database authentication with a suitable token/connection strategy rather than a hardcoded password; store any connection secret in Key Vault and reference it from App Service. `setup()` initializes checkpointer tables; coordinate schema migrations in production.

**Important:** `memory` mode is process-local and is lost on restart; it is unsuitable for multiple Azure App Service instances. Postgres preserves history across requests/restarts. The current implementation has no authentication or per-user thread authorization: add these before exposing conversation IDs to untrusted clients. Limit retention and implement deletion policies. Long conversations still require trimming or summarization to control tokens.

## Files changed
- `app/agent.py`: checkpointer injected into `create_agent`.
- `app/main.py`: lifecycle checkpointer, thread ID, resume history, turn-specific citations.
- `app/schemas.py`: optional request thread ID and response thread ID.
- `app/config.py`: memory backend settings.
- `requirements.txt`: PostgreSQL checkpointer dependency.

No `.env` file is included in this archive. Preserve your own local configuration and secrets.
