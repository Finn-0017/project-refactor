"""Model loading, LoRA configuration, and checkpoint utilities."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from peft import LoraConfig as PeftLoraConfig
from peft import TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from .utils import ensure_dir, load_json


@dataclass(frozen=True)
class LoRASettings:
    """LoRA settings used by the original project configuration files."""

    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_module: tuple[str, ...] = ("q_proj", "v_proj")
    uselora: bool = True

    @classmethod
    def from_json(cls, path: str | Path) -> "LoRASettings":
        data = load_json(path)
        return cls(
            lora_rank=int(data.get("lora_rank", data.get("r", 8))),
            lora_alpha=int(data.get("lora_alpha", 16)),
            lora_dropout=float(data.get("lora_dropout", 0.0)),
            lora_module=tuple(data.get("lora_module", ["q_proj", "v_proj"])),
            uselora=bool(data.get("uselora", True)),
        )


class CausalLMWithLoRA(torch.nn.Module):
    """A small wrapper around an instruction-tuned causal language model.

    The wrapper keeps the base-model interface explicit:

    - `forward(...)` uses the current model state, including LoRA adapters when enabled.
    - `base_model(...)` temporarily disables LoRA adapters and is used as the reference
      model for retain losses.
    - checkpoints save only trainable parameters, matching the original scripts.
    """

    def __init__(
        self,
        model_path: str,
        lora: LoRASettings,
        *,
        torch_dtype: str = "bfloat16",
        device_map: str | dict | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> None:
        super().__init__()
        self.model_path = model_path
        self.lora = lora
        dtype = getattr(torch, torch_dtype)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device_map,
        )

        if self.tokenizer.pad_token_id is None:
            # Most causal LMs used here do not have a separate padding token. Using EOS
            # keeps batching simple and avoids adding new vocabulary items.
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if lora.uselora:
            peft_config = PeftLoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=lora.lora_rank,
                lora_alpha=lora.lora_alpha,
                lora_dropout=lora.lora_dropout,
                target_modules=list(lora.lora_module),
            )
            self.model.add_adapter(peft_config)
            self.model.enable_adapters()
        else:
            # Full fine-tuning remains available, but the experiments discussed here use
            # LoRA. Keeping this branch explicit makes the training mode visible.
            for param in self.model.parameters():
                param.requires_grad = True

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        if attention_mask is None:
            # Evaluation often passes a single, unpadded chat prompt.  Some Llama
            # tokenizers use the EOS token as the padding token, and chat templates
            # contain EOS/EOT-like tokens inside real prompts.  A value-based mask
            # would hide valid prompt tokens, so the safe default is an all-ones mask.
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        return self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

    @contextmanager
    def adapters_disabled(self):
        """Temporarily disable adapters when computing reference-model outputs."""

        if self.lora.uselora and hasattr(self.model, "disable_adapters"):
            self.model.disable_adapters()
            try:
                yield
            finally:
                self.model.enable_adapters()
        else:
            yield

    def base_model(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        """Run the model with adapters disabled."""

        with self.adapters_disabled():
            return self.forward(input_ids, attention_mask=attention_mask)

    @torch.no_grad()
    def generate_text(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int = 128,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 0.9,
        use_base_model: bool = False,
    ) -> str:
        """Generate text after an already-tokenized chat prompt."""

        input_ids = input_ids.to(self.device)
        # Single-prompt generation should keep every token visible.  Training code
        # passes explicit masks for padded batches.
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        context = self.adapters_disabled() if use_base_model else _nullcontext()
        with context:
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output_ids[:, input_ids.size(1) :]
        return self.tokenizer.batch_decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def save_trainable_checkpoint(self, output_dir: str | Path) -> None:
        """Save only trainable tensors to `pytorch_model.pt`."""

        output_dir = ensure_dir(output_dir)
        state = {
            name: param.detach().cpu()
            for name, param in self.named_parameters()
            if param.requires_grad
        }
        torch.save(state, output_dir / "pytorch_model.pt")
        self.tokenizer.save_pretrained(output_dir)
        self.model.config.save_pretrained(output_dir)

    def load_trainable_checkpoint(self, checkpoint_path: str | Path, *, strict: bool = False) -> None:
        """Load a trainable-parameter checkpoint produced by this wrapper."""

        state = torch.load(checkpoint_path, map_location="cpu")
        self.load_state_dict(state, strict=strict)

    def trainable_parameters(self) -> Iterable[torch.nn.Parameter]:
        return (param for param in self.parameters() if param.requires_grad)


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False
