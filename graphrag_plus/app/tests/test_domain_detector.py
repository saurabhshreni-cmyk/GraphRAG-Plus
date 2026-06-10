"""Domain detector tests — word-boundary matching, not substrings."""

from graphrag_plus.app.domain.detector import detect_domain


def test_business_text_is_not_misclassified_by_substrings() -> None:
    # "continuation", "integration", "cancellation" contain the substring
    # "ion"; "Phase" contains "ph". None of these are chemistry signals.
    text = (
        "On 2024-01-15, Nova Dynamics acquired Orion Labs. "
        "Analyst Brief supports the claim that Orion Labs merged into Nova Dynamics. "
        "Another source contradicts the claim that Project Helios was canceled. "
        "Report Delta supports continuation under Phase II. "
        "Nova Dynamics supports Orion Labs integration roadmap."
    )
    assert detect_domain(text) == "general"


def test_chemistry_text_is_detected() -> None:
    text = (
        "The catalyst accelerates the reaction between the acid and the base. "
        "Each molecule shares an electron pair through a covalent bond, and the "
        "solvent dissolves the solute to form a solution with neutral pH."
    )
    assert detect_domain(text) == "chemistry"


def test_machine_learning_text_is_detected() -> None:
    text = (
        "We trained a neural network with gradient descent and backpropagation. "
        "The transformer uses attention layers, dropout regularization, and a "
        "softmax loss function to avoid overfitting on the training set."
    )
    assert detect_domain(text) == "machine_learning"


def test_empty_text_falls_back_to_general() -> None:
    assert detect_domain("") == "general"
    assert detect_domain("   \n\t ") == "general"


def test_short_ambiguous_text_falls_back_to_general() -> None:
    assert detect_domain("Hello world, this is a plain note.") == "general"
