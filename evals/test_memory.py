def test_conversation_memory(chat, thread_id):
    first = chat(
        "We are discussing Illinois. "
        "What is its capital?",
        thread_id
    )

    assert first["thread_id"] == thread_id
    assert "springfield" in first["response"].lower()

    second = chat(
        "Which state were we just discussing?",
        thread_id
    )

    assert "illinois" in second["response"].lower()