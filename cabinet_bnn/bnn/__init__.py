"""Cabinet-BNN：二值激活 + HSH-64 权重码。"""
from .model import (
    BNNConfig, BNNWordLM, CodeWeightLinear, WeightCodeConfig, WeightCodeTable,
    binary_activation, build_model, count_params, ste_sign,
)
from .corpus import CharCorpus, build_char_corpus, make_batches, train_val_split

__all__ = [
    "BNNConfig", "BNNWordLM", "CodeWeightLinear", "WeightCodeConfig",
    "WeightCodeTable", "binary_activation", "build_model", "count_params",
    "ste_sign", "CharCorpus", "build_char_corpus", "make_batches", "train_val_split",
]
