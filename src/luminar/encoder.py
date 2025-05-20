from typing import Callable, Self

import torch
from torch import Tensor, nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import (
    BatchEncoding,
    EncodedInput,
    PreTrainedTokenizerBase,
)

type OneOrTwoDevices = (
    str | torch.device | tuple[str, str] | tuple[torch.device, torch.device]
)


class LuminarEncoder:
    def __init__(
        self,
        model_name_or_path: str = "gpt2",
        max_len: int = 1024,
        low_memory: bool = False,
        device: OneOrTwoDevices = ("cuda" if torch.cuda.is_available() else "cpu"),
        get_lm_head_from_model_fn: Callable[[PreTrainedModel], nn.Linear] | None = None,
    ):
        """Initialize the LuminarEncoder.
        This class is designed to encode text sequences using a pre-trained language model.
        It uses the Hugging Face `transformers` library to load the model and tokenizer.
        The model is used to compute intermediate likelihoods for each token in the input
        sequence across all hidden states.

        Computation may be sped-up by providing a second device for the LM head.
        You can also use the `device` setter to move the model and LM head to different devices.

        Args:
            model_name_or_path (str, optional): Model name or path, passed to `AutoModelForCausalLM.from_pretrained`.
                Defaults to "gpt2".
            max_len (int, optional): Determines the maximum sequence length for the input sequences.
                If the input sequence is longer than this length, it will be truncated.
                Defaults to 1024.
            low_memory (bool, optional): Flag that determines whether to use a fast, but memory expensive
                implementation, or a slower, but more memory efficient implementation.
                Defaults to False.
            device (OneOrTwoDevices, optional): Can be a single device, or a tuple of two devices.
                If a single device is provided, both the model and the LM head will be moved to that device.
                If two devices are provided, the model will be moved to the first device, and the LM head
                will be moved to the second device. Defaults to "cuda" if available else "cpu".
            get_lm_head_from_model_fn (Callable[[PreTrainedModel], nn.Linear] | None, optional): A callable
                that extracts the LM head (a `torch.nn.Linear` layer) from the model.
                By default, the LM head is constructed from the model's output embeddings.
                See `get_lm_head_from_output_embeddings(PreTrainedModel, torch.device, bool)` for more details.
        """
        self._max_len = max_len
        self._low_memory = low_memory

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_name_or_path
        )
        if not hasattr(self.tokenizer, "pad_token") or self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id: int = self.tokenizer.pad_token_id  # type: ignore

        device_model, device_lm_head = extract_device_pairs(device)

        self._model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map={"": device_model},
        )
        self._model.eval()

        self._lm_head: nn.Linear = (
            get_lm_head_from_model_fn(self._model)
            if get_lm_head_from_model_fn
            else get_lm_head_from_output_embeddings(
                self._model,
                device_lm_head,
                copy_weights=device_lm_head != device_model,
            )
        )
        self._lm_head.eval()
        self.device_model = device_model
        self.device_lm_head = device_lm_head

    @property
    def device_model(self) -> torch.device:
        return self._device_model

    @device_model.setter
    def device_model(self, device: str | torch.device):
        self._device_model = torch.device(device)
        self._model.to(self._device_model)  # type: ignore

    @property
    def device_lm_head(self) -> torch.device:
        return self._device_lm_head

    @device_lm_head.setter
    def device_lm_head(self, device: str | torch.device):
        self._device_lm_head = torch.device(device)
        self._lm_head.to(self._device_lm_head)  # type: ignore

    @property
    def device(self) -> tuple[torch.device, torch.device]:
        return self.device_model, self.device_lm_head

    @device.setter
    def device(self, device: OneOrTwoDevices):
        device_a, device_b = extract_device_pairs(device)

        self.device_model = device_a
        self.device_lm_head = device_b

    def to(self, *device: str) -> Self:
        """Move the model to the specified device(s).

        Args:
            device (OneOrTwoDevices): The device(s) to move the model to. This can be a single
                device, or a tuple of two devices. If a single device is provided, both the
                model and the lm_head will be moved to that device. If two devices are provided,
                the model will be moved to the first device, and the lm_head will be moved to
                the second device.
        """
        self.device = device  # type: ignore
        return self

    def tokenize(self, texts: str | list[str]) -> BatchEncoding:
        """Convenience method to tokenize the input texts. This is a wrapper around the
        `transformers` tokenizer. It handles padding and truncation of the input sequences.

        Args:
            texts (str | list[str]): text or texts to tokenize.

        Returns:
            BatchEncoding: The resulting encoding object with the lengths of each input sequence.
        """
        return self.tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=self._max_len,
            return_length=True,
            add_special_tokens=True,
        )

    def process(
        self, inputs: dict[str, EncodedInput]
    ) -> dict[str, list[list[list[float]]]]:
        """Process the input sequences and compute the intermediate likelihoods.
        Designed to be used with the `datasets` library.

        Args:
            inputs (dict[str, EncodedInput]): The encoded input sequences. This should be a dictionary
                containing the `input_ids` and `attention_mask`.

        Returns:
            dict[str, list[list[list[float]]]]: A dictionary containing a single key `features`
                with intermediate likelihoods for each token in the input sequence across all hidden states,
                converted into a list using `Tensor.tolist()`.
        """
        return {
            "features": [
                il.tolist()
                for il in self.encode(
                    {
                        "input_ids": inputs["input_ids"],
                        "attention_mask": inputs["attention_mask"],
                    }
                )
            ]
        }

    @torch.inference_mode()
    def encode(self, batch: dict[str, EncodedInput]) -> list[Tensor]:
        """Encode a batch of input sequences and compute the intermediate likelihoods.

        Args:
            batch (dict[str, EncodedInput]): The encoded batch of input sequences. This should
                be a dictionary containing the `input_ids` and `attention_mask`.

        Returns:
            list[Tensor]: A list of tensors containing the intermediate likelihoods for each
                token in the input sequence across all hidden states.
        """
        encoding = self.tokenizer.pad(batch, return_tensors="pt").to(self._device_model)

        batch_hidden_states = self.forward(encoding.input_ids, encoding.attention_mask)

        intermediate_likelihoods = [
            self.compute_intermediate_likelihoods(input_ids, hidden_states)
            for input_ids, hidden_states in zip(encoding.input_ids, batch_hidden_states)
        ]

        return intermediate_likelihoods

    @torch.inference_mode()
    def forward(
        self, input_ids: Tensor, attention_mask: Tensor, *_, **__
    ) -> list[Tensor]:
        """Forward pass through the model to get the hidden states.

        Args:
            input_ids (Tensor): The input ids of the encoded sequence.
            attention_mask (Tensor): The attention mask of the encoded sequence.

        Returns:
            list[Tensor]: A list of tensors containing the hidden states for each layer in the model,
                with one tensor per input sequence.
        """
        outputs = self._model(
            input_ids=input_ids.to(self._device_model),
            attention_mask=attention_mask.to(self._device_model),
            output_hidden_states=True,
        )

        # unpack hidden states to get one list of tensors per input sequence,
        # instead of one hidden state per layer in the model
        return [torch.stack(hs, dim=1) for hs in zip(*outputs.hidden_states)]

    @torch.inference_mode()
    def compute_intermediate_likelihoods(
        self, input_ids: Tensor, hidden_states: Tensor
    ) -> Tensor:
        """Compute the intermediate likelihoods for the input tokens.

        Args:
            input_ids (Tensor): The input ids of the encoded sequence.
            hidden_states (Tensor): The stacked hidden states as a a Tensor of shape
                (seq_length, model_depth, hidden_size)

        Returns:
            Tensor: A single Tensor of shape (seq_length, model_depth) containing the
                likelihoods for each token in the input sequence across all hidden states.

        Note:
            The `low_memory` flag determines whether to use a fast, but memory expensive
            implementation, or a slower, but more memory efficient implementation.
        """
        if not self._low_memory:
            return self.compute_intermediate_likelihoods_fast(input_ids, hidden_states)
        else:
            return self.compute_intermediate_likelihoods_slow(input_ids, hidden_states)

    @torch.inference_mode()
    def compute_intermediate_likelihoods_fast(
        self, input_ids: Tensor, hidden_states: Tensor
    ) -> Tensor:
        """Compute the intermediate likelihoods for the input tokens in a fast, but memory
        expensive way. Used when `low_memory` is set to False.
        This version is implemented as a single forward pass through the models' LM head.

        Args:
            input_ids (Tensor): The input ids of the encoded sequence.
            hidden_states (Tensor): The stacked hidden states as a a Tensor of shape
                (seq_length, model_depth, hidden_size)

        Returns:
            Tensor: A single Tensor of shape (seq_length, model_depth) containing the
                likelihoods for each token in the input sequence across all hidden states.

        Note:
            This may require a lot of memory, depending on the size of the output embeddings!
        """
        seq_length: int = int(input_ids[1:].ne(self.pad_token_id).sum().item())
        seq_length = min(seq_length, self._max_len)

        # remove the first token and padding tokens from the input_ids
        labels = input_ids[1 : 1 + seq_length].to(self.device_lm_head)

        # calculate the likelihoods for each hidden state
        # and retain the likelihoods for the input tokens
        intermediate_likelihoods = (
            # get layer logits
            self._lm_head(hidden_states.to(self.device_lm_head))
            # calculate likelihoods
            .softmax(-1)[
                # get likelihoods of input tokens
                torch.arange(seq_length), :, labels
            ]
        )
        return intermediate_likelihoods.cpu()

    @torch.inference_mode()
    def compute_intermediate_likelihoods_slow(
        self, input_ids: Tensor, hidden_states: Tensor
    ) -> Tensor:
        """Compute the intermediate likelihoods for the input token in a slower, but more
        memory efficient way. Used when `low_memory` is set to True.
        This version is implemented as multiple forward passes through the models' LM head,
        one for each hidden state.

        Args:
            input_ids (Tensor): The input ids of the encoded sequence.
            hidden_states (Tensor): The stacked hidden states as a a Tensor of shape
                (seq_length, model_depth, hidden_size)

        Returns:
            Tensor: A single Tensor of shape (seq_length, model_depth) containing the
                likelihoods for each token in the input sequence across all hidden states.

        Note:
            This is quite slow, as we are running the LM head forward pass in a for-loop here.
        """
        seq_length: int = int(input_ids[1:].ne(self.pad_token_id).sum().item())
        seq_length = min(seq_length, self._max_len)

        labels = input_ids[1 : 1 + seq_length].to(self.device_lm_head)

        intermediate_likelihoods = []
        for hs in hidden_states:
            hs: Tensor = hs[:seq_length].to(self.device_lm_head)
            il = (
                # get layer logits
                self._lm_head(hs)
                # calculate likelihoods
                .softmax(-1)
                # get likelihoods of input tokens
                .gather(-1, labels)
                .squeeze(-1)
                .cpu()
            )
            del hs

            intermediate_likelihoods.append(il)

        # stack intermediate likelihoods to get tensor of shape (feature_dim, num_layers)
        return torch.stack(intermediate_likelihoods, dim=1)


def extract_device_pairs(device: OneOrTwoDevices) -> tuple[torch.device, torch.device]:
    if isinstance(device, (str, torch.device)):
        device_a = device
        device_b = device
    elif isinstance(device, (tuple, list)):
        if len(device) == 1:
            (device_a,) = device
            device_b = device_a
        else:
            device_a, device_b = device
    else:
        raise ValueError(
            f"Invalid device: {type(device).__name__}({device}). Must be a str, torch.device, or a tuple of two devices."
        )

    device_a = torch.device(device_a)
    device_b = torch.device(device_b)

    return device_a, device_b


def get_lm_head_from_output_embeddings(
    model: PreTrainedModel, device: torch.device, copy_weights: bool = False
) -> nn.Linear:
    if (
        hasattr(model, "get_output_embeddings")
        and (output_embeddings := model.get_output_embeddings()) is not None
    ):
        hidden_size = model.config.hidden_size
        n_feats_in, n_feats_out = tuple(output_embeddings.weight.shape)  # type: ignore
        if n_feats_in != hidden_size:
            if n_feats_out != hidden_size:
                raise ValueError(
                    "Failed to determine the input size of the LM head: "
                    f"hidden_size={hidden_size}, but "
                    f"n_feats_in={n_feats_in} & n_feats_out={n_feats_out}"
                )
            n_feats_in, n_feats_out = n_feats_out, n_feats_in

        lm_head_requires_bias = output_embeddings.bias is not None

        lm_head: nn.Linear = nn.Linear(
            n_feats_in,
            n_feats_out,
            bias=lm_head_requires_bias,
            device=device,
        )
        lm_head.requires_grad_(False)

        with torch.no_grad():
            if copy_weights:
                lm_head.weight.copy_(output_embeddings.weight)  # type: ignore
                if lm_head_requires_bias:
                    lm_head.bias.copy_(output_embeddings.bias)  # type: ignore
            else:
                lm_head.weight = output_embeddings.weight  # type: ignore
                if lm_head_requires_bias:
                    lm_head.bias = output_embeddings.bias  # type: ignore
            # elif hasattr(model, "lm_head"):
            #     lm_head: nn.Linear = model.lm_head  # type: ignore
            # elif hasattr(model.model, "lm_head"):
            #     lm_head: nn.Linear = model.model.lm_head  # type: ignore

        return lm_head
    else:
        raise ValueError("Could not find lm_head in model")
