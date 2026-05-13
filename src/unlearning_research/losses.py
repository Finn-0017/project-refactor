"""Loss functions for DF-MCQ and WHP-style training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def uniform_choice_cross_entropy(choice_logits: torch.Tensor) -> torch.Tensor:
    """Cross-entropy from a uniform target distribution to model choice logits.

    This loss encourages all answer letters to be equally likely. It is equivalent to
    KL(uniform || model_choice_distribution) up to a constant.
    """

    log_probs = torch.log_softmax(choice_logits, dim=-1)
    return -log_probs.mean(dim=-1).mean()


def model_to_uniform_kl(choice_logits: torch.Tensor) -> torch.Tensor:
    """KL(model_choice_distribution || uniform).

    This direction matches the notation often used to describe distribution flattening.
    Both this loss and `uniform_choice_cross_entropy` have the same optimum: a flat
    choice distribution. The cross-entropy version is usually more stable in practice.
    """

    probs = torch.softmax(choice_logits, dim=-1)
    log_probs = torch.log_softmax(choice_logits, dim=-1)
    log_uniform = -torch.log(torch.tensor(choice_logits.size(-1), device=choice_logits.device))
    return (probs * (log_probs - log_uniform)).sum(dim=-1).mean()


def reference_choice_cross_entropy(
    student_choice_logits: torch.Tensor,
    reference_choice_probs: torch.Tensor,
) -> torch.Tensor:
    """Preserve the original model's choice distribution on retain MCQs."""

    log_probs = torch.log_softmax(student_choice_logits, dim=-1)
    target = reference_choice_probs.detach()
    return -(target * log_probs).sum(dim=-1).mean()


def causal_lm_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Standard next-token cross-entropy for causal language modeling."""

    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.size(-1)),
        shifted_labels.view(-1),
        ignore_index=ignore_index,
    )


def gradient_l2_norm(model: torch.nn.Module) -> float:
    """Return the L2 norm of current gradients over trainable parameters."""

    total = 0.0
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            total += float(param.grad.detach().norm(2).item() ** 2)
    return total ** 0.5
