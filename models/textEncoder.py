import torch
from torch import Tensor
from transformers import T5EncoderModel, T5Tokenizer
from beartype import beartype
from jaxtyping import jaxtyped, Float, Bool
from beartype.typing import List, Union, Tuple


class T5TextEncoder(torch.nn.Module):
    def __init__(self, modelName: str = "google-t5/t5-11b") -> None:
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(modelName)
        self.textEncoder = T5EncoderModel.from_pretrained(modelName)
        self.textEncoder.eval()

    @torch.inference_mode(True)
    @jaxtyped(typechecker=beartype)
    def forward(
        self, text: Union[str, List[str]]
    ) -> Tuple[Float[Tensor, "batch seq embed"], Bool[Tensor, "batch seq"]]:

        tokens = self.tokenizer(  # pyrefly:ignore
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512,
            return_length=True,
            return_offsets_mapping=False,
        )

        inputIds = tokens["input_ids"].to(self.textEncoder.device)
        attentionMask = tokens["attention_mask"].to(self.textEncoder.device)

        output = self.textEncoder(input_ids=inputIds, attention_mask=attentionMask)
        maskedHiddenStates = output.last_hidden_state * attentionMask.unsqueeze(-1)
        return maskedHiddenStates, attentionMask.bool()
