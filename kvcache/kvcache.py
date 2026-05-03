# kv cache implementation from scratch for fun


import math
import torch
import torch.nn.functional as torch_functional

class KVCache:
    """
    stores keys/values for autoregressive decoding

    notably does not touch attention at all
    """
    def __init__(
        self,
        maximum_sequence_length: int,
        head_dimension: int,
        device: str="cpu",
        dtype: torch.dtype=torch.float32,
    ):
        """
        prealloc contiguous memory

        keys: [max seq len, head dim]
        values: [max seq len, head dim]
        """

        # save inputs:
        self.maximum_sequence_length = maximum_sequence_length
        self.head_dimension = head_dimension
        self.device=device
        self.dtype=dtype
        

        self.key_cache = torch.zeros(
            maximum_sequence_length,
            head_dimension,
            device=device,
            dtype=dtype,
        )

        self.value_cache = torch.zeros(
            maximum_sequence_length,
            head_dimension,
            device=device,
            dtype=dtype,
        )