from unlearning_research.eval_score import _first_letter, _first_yes_no, _obfuscation_letters, score_suite


def test_label_extraction():
    assert _first_letter('B. The answer is here') == 'B'
    assert _first_yes_no('No, that is not correct') == 'No'


def test_obfuscation_letter_mapping():
    choices = {'A': 'Paris', 'B': 'Berlin'}
    assert _obfuscation_letters('A', choices) == ['A']
    assert _obfuscation_letters('Berlin', choices) == ['B']


def test_score_suite():
    predictions = {
        'metadata': {},
        'mcq': {'Alice': [{'ref': 'A', 'pred_letter': 'A', 'generated_letter': 'A', 'entropy': 1.0, 'normalized_entropy': 0.5, 'p_correct': 0.7, 'p_obfuscation': 0.1, 'is_refused': False}]},
        'yes_no_reference': {'Alice': [{'ref': 'Yes', 'pred_label': 'No', 'generated_label': 'No', 'p_yes': 0.2, 'entropy': 0.5, 'normalized_entropy': 0.7, 'is_refused': False}]},
        'open_ended': {'Alice': [{'ref': 'x', 'pred': 'I do not know.', 'rougeL_recall': 0.0, 'is_refused': True}]},
    }
    summary = score_suite(predictions)
    assert summary['mcq']['accuracy'] == 1.0
    assert summary['yes_no_reference']['accuracy'] == 0.0
    assert summary['open_ended']['refusal_rate'] == 1.0
