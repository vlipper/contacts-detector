from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st
import torch
from transformers import AutoTokenizer

if TYPE_CHECKING:
    from torch.jit import ScriptModule
    from transformers import PreTrainedTokenizerFast


@st.cache_resource
def load_model(model_path: str) -> ScriptModule:
    model = torch.jit.load(model_path)
    return model


@st.cache_resource
def load_tokenizer(tokenizer_path: str) -> PreTrainedTokenizerFast:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return tokenizer
