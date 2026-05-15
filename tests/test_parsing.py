from unlearning_research.parsing import extract_mcq_letter


def test_extract_mcq_direct_letter():
    assert extract_mcq_letter('B', ('A', 'B', 'C')) == 'B'
    assert extract_mcq_letter('B. The option text', ('A', 'B', 'C')) == 'B'


def test_extract_mcq_phrase():
    assert extract_mcq_letter('The correct answer is C.', ('A', 'B', 'C')) == 'C'


def test_extract_mcq_choice_text():
    choices = {'A': 'Paris', 'B': 'Berlin'}
    assert extract_mcq_letter('Berlin', ('A', 'B'), choices) == 'B'
