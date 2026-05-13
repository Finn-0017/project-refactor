import math

import torch

from unlearning_research.choice import entropy_from_probs, normalized_choice_probs
from unlearning_research.losses import uniform_choice_cross_entropy


def test_entropy_of_uniform_five_choice_distribution():
    probs = torch.full((1, 5), 0.2)
    entropy = entropy_from_probs(probs, normalized=False)
    assert torch.allclose(entropy, torch.tensor([math.log(5.0)]), atol=1e-6)


def test_normalized_entropy_of_uniform_distribution_is_one():
    probs = torch.full((1, 5), 0.2)
    entropy = entropy_from_probs(probs, normalized=True)
    assert torch.allclose(entropy, torch.tensor([1.0]), atol=1e-6)


def test_choice_softmax_normalizes_over_choices_only():
    logits = torch.tensor([[10.0, 10.0, 10.0, 10.0, 10.0]])
    probs = normalized_choice_probs(logits)
    assert torch.allclose(probs.sum(dim=-1), torch.tensor([1.0]))


def test_uniform_choice_loss_is_smallest_for_uniform_logits():
    uniform_logits = torch.zeros(2, 5)
    sharp_logits = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0], [0.0, 10.0, 0.0, 0.0, 0.0]])
    assert uniform_choice_cross_entropy(uniform_logits) < uniform_choice_cross_entropy(sharp_logits)
