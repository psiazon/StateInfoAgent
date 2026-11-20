def test_company_search_returns_citations(chat):
    result = chat(
        "Search the company documents and identify "
        "one company described in the indexed PDFs. "
        "Cite the document used."
    )

    tools_used = result.get("tools_used", [])
    citations = result.get("citations", [])

    assert "search_company_documents" in tools_used

    assert any(
        citation.get("source") == "Azure AI Search"
        and citation.get("title")
        for citation in citations
    ), "Expected at least one Azure AI Search citation"