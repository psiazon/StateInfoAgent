def test_state_question_uses_mcp(chat):
    result = chat(
        "Using state records, what is "
        "the capital of Illinois?"
    )

    tools_used = result.get("tools_used", [])

    assert tools_used, (
        "Expected at least one tool call"
    )

    assert "search_company_documents" not in tools_used, (
        "A state-specific question should use MCP, "
        "not company document search"
    )