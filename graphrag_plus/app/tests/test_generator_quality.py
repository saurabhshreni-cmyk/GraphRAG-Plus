"""Regression tests for answer quality upgrades."""

from graphrag_plus.app.generation.generator import AnswerGenerator, _is_clean_sentence
from graphrag_plus.app.ingestion.chunker import _fixed_window


def _evidence(*items: dict[str, str]) -> list[dict[str, object]]:
    return [dict(item) for item in items]


def test_fixed_window_aligns_chunks_to_word_boundaries() -> None:
    text = (
        "Constant Error Carousel units help preserve gradients across long sequences "
        "without splitting key phrases in the middle of words."
    )

    windows = _fixed_window(text, chunk_size=36, chunk_overlap=8)

    assert len(windows) >= 2
    assert all(not chunk.startswith("ant ") for chunk, _, _ in windows)
    assert all(chunk == chunk.strip() for chunk, _, _ in windows)


def test_definition_answer_prefers_clean_definition_over_fragment() -> None:
    generator = AnswerGenerator(llm_enabled=False)
    evidence = _evidence(
        {
            "id": "chunk-1",
            "source_id": "doc-1",
            "snippet": "",
            "full_text": (
                "Error Carousel units help preserve gradients, and LSTM deals with long-range "
                "dependencies in sequence models."
            ),
        },
        {
            "id": "chunk-2",
            "source_id": "doc-1",
            "snippet": "",
            "full_text": (
                "Long short-term memory is a type of recurrent neural network architecture "
                "designed to model sequences and preserve information over time."
            ),
        },
    )

    answer, used_llm, llm_failed = generator.generate(
        "What is LSTM?",
        evidence,
        confidence=0.4,
        answer_threshold=0.7,
        intent="definition",
    )

    assert used_llm is False
    assert llm_failed is False
    assert answer == (
        "LSTM is a type of recurrent neural network architecture designed to model "
        "sequences and preserve information over time."
    )


def test_clean_sentence_rejects_fragments_without_complete_sentence_shape() -> None:
    assert not _is_clean_sentence("Error Carousel (CEC) units help preserve gradients.")
    assert not _is_clean_sentence("s interpreted as saying that the particles share a state.")
    assert not _is_clean_sentence("Quantum entanglement physical phenomenon.")
    assert not _is_clean_sentence("Quantum entanglement is a physical phenomenon")
    assert _is_clean_sentence(
        "Quantum entanglement is a physical phenomenon involving shared quantum states."
    )


def test_definition_answer_requires_definition_pattern() -> None:
    generator = AnswerGenerator(llm_enabled=False)
    evidence = _evidence(
        {
            "id": "chunk-1",
            "source_id": "doc-1",
            "snippet": "",
            "full_text": (
                "LSTM uses gates to control information flow across long sequences. "
                "Long short-term memory is a type of recurrent neural network architecture."
            ),
        }
    )

    answer, _, _ = generator.generate(
        "What is LSTM?",
        evidence,
        confidence=0.4,
        answer_threshold=0.7,
        intent="definition",
    )

    assert answer == "LSTM is a type of recurrent neural network architecture."


def test_quantum_definition_rejects_leading_fragment() -> None:
    generator = AnswerGenerator(llm_enabled=False)
    evidence = _evidence(
        {
            "id": "chunk-1",
            "source_id": "doc-1",
            "snippet": "",
            "full_text": (
                "s interpreted as saying that the particles share a state. "
                "Quantum entanglement is a physical phenomenon involving shared quantum states."
            ),
        }
    )

    answer, _, _ = generator.generate(
        "What is quantum entanglement?",
        evidence,
        confidence=0.4,
        answer_threshold=0.7,
        intent="definition",
    )

    assert answer == "Quantum entanglement is a physical phenomenon involving shared quantum states."


def test_explanation_answer_rejects_broken_sentences() -> None:
    generator = AnswerGenerator(llm_enabled=False)
    evidence = _evidence(
        {
            "id": "chunk-1",
            "source_id": "doc-1",
            "snippet": "",
            "full_text": (
                "ing gradients are frequently mentioned in sequence learning. "
                "LSTM uses gates to control information flow across long sequences. "
                "LSTM maintains cell state so relevant information can persist over time."
            ),
        }
    )

    answer, _, _ = generator.generate(
        "How does LSTM preserve information?",
        evidence,
        confidence=0.4,
        answer_threshold=0.7,
        intent="explanation",
    )

    assert "ing gradients" not in answer
    assert "LSTM maintains cell state so relevant information can persist over time." in answer
    assert "LSTM uses gates to control information flow across long sequences." in answer


def test_factual_fallback_returns_only_complete_sentences() -> None:
    generator = AnswerGenerator(llm_enabled=False)
    evidence = _evidence(
        {
            "id": "chunk-1",
            "source_id": "doc-1",
            "snippet": "",
            "full_text": (
                "ant Error Carousel units preserve gradients. "
                "LSTM uses gates to control information flow across long sequences."
            ),
        }
    )

    answer, _, _ = generator.generate(
        "Which model uses gates?",
        evidence,
        confidence=0.4,
        answer_threshold=0.7,
        intent="factual",
    )

    assert answer == "LSTM uses gates to control information flow across long sequences."


def test_factual_answer_accepts_business_event_verbs() -> None:
    generator = AnswerGenerator(llm_enabled=False)
    evidence = _evidence(
        {
            "id": "chunk-1",
            "source_id": "doc-1",
            "snippet": "",
            "full_text": "On 2024-01-15, Nova Dynamics acquired Orion Labs.",
        }
    )

    answer, _, _ = generator.generate(
        "What did Nova Dynamics acquire?",
        evidence,
        confidence=0.8,
        answer_threshold=0.7,
        intent="factual",
    )

    assert answer == "On 2024-01-15, Nova Dynamics acquired Orion Labs."


def test_factual_answer_accepts_contradiction_verbs() -> None:
    generator = AnswerGenerator(llm_enabled=False)
    evidence = _evidence(
        {
            "id": "chunk-1",
            "source_id": "doc-1",
            "snippet": "",
            "full_text": "Report Delta contradicts the claim that Project Helios was canceled.",
        }
    )

    answer, _, _ = generator.generate(
        "Which source contradicts Project Helios cancellation?",
        evidence,
        confidence=0.8,
        answer_threshold=0.7,
        intent="factual",
    )

    assert answer == "Report Delta contradicts the claim that Project Helios was canceled."


def test_llm_context_filters_broken_and_duplicate_sentences() -> None:
    evidence = _evidence(
        {
            "id": "chunk-1",
            "source_id": "doc-1",
            "snippet": (
                "s interpreted as saying that particles share a state. "
                "Quantum entanglement is a physical phenomenon involving shared quantum states."
            ),
        },
        {
            "id": "chunk-2",
            "source_id": "doc-1",
            "snippet": "Quantum entanglement is a physical phenomenon involving shared quantum states.",
        },
    )

    context = AnswerGenerator._build_context(evidence)

    assert "s interpreted" not in context
    assert context.count("Quantum entanglement is a physical phenomenon") == 1


def test_list_answer_filters_clause_fragments_and_keeps_real_methods() -> None:
    generator = AnswerGenerator(llm_enabled=False)
    evidence = _evidence(
        {
            "id": "chunk-1",
            "source_id": "doc-1",
            "snippet": "",
            "full_text": (
                "Common depreciation methods are straight-line method, declining balance method, "
                "sum-of-the-years'-digits method, and units-of-production method. "
                "Depreciation reflects asset wear and is not based on time alone."
            ),
        }
    )

    answer, _, _ = generator.generate(
        "Types of depreciation?",
        evidence,
        confidence=0.4,
        answer_threshold=0.7,
        intent="list",
    )

    assert "1. straight-line method" in answer.lower()
    assert "2. declining balance method" in answer.lower()
    assert "3. sum-of-the-years'-digits method" in answer.lower()
    assert "4. units-of-production method" in answer.lower()
    assert "asset wear" not in answer.lower()
    assert "not based on time" not in answer.lower()


def test_list_answer_rejects_generic_clause_noun_phrases() -> None:
    generator = AnswerGenerator(llm_enabled=False)
    evidence = _evidence(
        {
            "id": "chunk-1",
            "source_id": "doc-1",
            "snippet": "",
            "full_text": (
                "Methods include asset wear, type of taxpayer, not based on time, "
                "straight-line method, and declining balance method."
            ),
        }
    )

    answer, _, _ = generator.generate(
        "Types of depreciation?",
        evidence,
        confidence=0.4,
        answer_threshold=0.7,
        intent="list",
    )

    assert "straight-line method" in answer.lower()
    assert "declining balance method" in answer.lower()
    assert "asset wear" not in answer.lower()
    assert "type of taxpayer" not in answer.lower()
