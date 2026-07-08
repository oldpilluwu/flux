import transformers
from torch import Tensor, nn
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer


class HFEmbedder(nn.Module):
    def __init__(self, version: str, max_length: int, **hf_kwargs):
        super().__init__()
        self.is_clip = version.startswith("openai")
        self.max_length = max_length
        self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"

        # transformers renamed torch_dtype -> dtype in v5 and silently ignores
        # the unknown name, which would load these fp32 checkpoints at full
        # precision (~19 GB host+GPU for T5-XXL instead of ~9.5 GB). Normalize
        # to whichever name the installed version honors, so the weights are
        # loaded (not post-cast) at the requested dtype.
        requested_dtype = hf_kwargs.pop("torch_dtype", None) or hf_kwargs.pop("dtype", None)
        if requested_dtype is not None:
            major = int(transformers.__version__.split(".")[0])
            hf_kwargs["dtype" if major >= 5 else "torch_dtype"] = requested_dtype

        if self.is_clip:
            self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(version, max_length=max_length)
            self.hf_module: CLIPTextModel = CLIPTextModel.from_pretrained(version, **hf_kwargs)
        else:
            self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(version, max_length=max_length)
            self.hf_module: T5EncoderModel = T5EncoderModel.from_pretrained(version, **hf_kwargs)

        # belt-and-suspenders: if the kwarg still didn't stick, cast in place
        if requested_dtype is not None and next(self.hf_module.parameters()).dtype != requested_dtype:
            self.hf_module = self.hf_module.to(requested_dtype)

        self.hf_module = self.hf_module.eval().requires_grad_(False)

    def forward(self, text: list[str]) -> Tensor:
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )

        outputs = self.hf_module(
            input_ids=batch_encoding["input_ids"].to(self.hf_module.device),
            attention_mask=None,
            output_hidden_states=False,
        )
        return outputs[self.output_key].bfloat16()
