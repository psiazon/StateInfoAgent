# StateInfoAgent — Azure AI Search RAG extension

This adds an optional `search_company_documents` LangChain tool to the existing MCP-backed agent.

## Local
1. Create a venv and `pip install -r requirements.txt`.
2. Copy `env.example` to `.env` and fill in your existing Foundry/MCP endpoints and Azure Search endpoint.
3. Sign in with `az login` or another supported DefaultAzureCredential developer login.
4. Grant your developer identity **Search Index Data Reader** on the Search service.
5. Run `python -m uvicorn app.main:app --reload --port 8000`.
6. Check `/api/agent/tools` for `search_company_documents`; then POST `/api/agent/chat` with `{"message":"Which companies manufacture heavy equipment?"}`.

## Azure Web App
- Enable system-assigned managed identity.
- Assign **Search Index Data Reader** to the Web App identity on Azure AI Search.
- Add `AZURE_SEARCH_*` application settings (no Search API key).
- Keep existing Foundry/MCP configuration and permissions.
- Ensure the Azure AI Search index vectorizer has access to its embedding deployment.

## Limitations
- Uses hybrid BM25 + vector retrieval via the existing index's query-time vectorizer.
- This index exposes `chunk`, `title`, `text_vector`, but not structured `company_name` or `state` fields.
- Similarity retrieval is **not exhaustive** for queries such as 'all states where Caterpillar operates'. Add filterable structured company/state records for reliable enumeration.
- PDF citations include retrieved title/snippet; source URLs are omitted because this index does not expose a retrievable source URL field.
- Network/Entra permissions and live integration are not tested in this package.
