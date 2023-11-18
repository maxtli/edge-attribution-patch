# %%
from IPython import get_ipython
ipython = get_ipython()
if ipython is not None:
    ipython.magic("%load_ext autoreload")
    ipython.magic("%autoreload 2")

import os
import sys
sys.path.append('../Automatic-Circuit-Discovery/')
sys.path.append('..')
import torch
import re

import acdc
from utils.prune_utils import get_3_caches, split_layers_and_heads
from acdc.TLACDCExperiment import TLACDCExperiment
from acdc.acdc_utils import TorchIndex, EdgeType
import numpy as np
import torch as t
from torch import Tensor
import einops
import itertools

from transformer_lens import HookedTransformer, ActivationCache

import tqdm.notebook as tqdm
import plotly
from rich import print as rprint
from rich.table import Table

from jaxtyping import Float, Bool
from typing import Callable, Tuple, Union, Dict, Optional

device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
print(f'Device: {device}')

# %% [markdown]
# # Model Setup

# %%
model = HookedTransformer.from_pretrained(
    'gpt2-small',
    center_writing_weights=False,
    center_unembed=False,
    fold_ln=False,
    device=device,
)
model.set_use_hook_mlp_in(True)
model.set_use_split_qkv_input(True)
model.set_use_attn_result(True)

# %% [markdown]
# # Dataset Setup

# %%
from ioi_dataset import IOIDataset, format_prompt, make_table
N = 25
clean_dataset = IOIDataset(
    prompt_type='mixed',
    N=N,
    tokenizer=model.tokenizer,
    prepend_bos=False,
    seed=1,
    device=device
)
corr_dataset = clean_dataset.gen_flipped_prompts('ABC->XYZ, BAB->XYZ')

make_table(
  colnames = ["IOI prompt", "IOI subj", "IOI indirect obj", "ABC prompt"],
  cols = [
    map(format_prompt, clean_dataset.sentences),
    model.to_string(clean_dataset.s_tokenIDs).split(),
    model.to_string(clean_dataset.io_tokenIDs).split(),
    map(format_prompt, clean_dataset.sentences),
  ],
  title = "Sentences from IOI vs ABC distribution",
)

# %% [markdown]
# # Metric Setup

# %%
def ave_logit_diff(
    logits: Float[Tensor, 'batch seq d_vocab'],
    ioi_dataset: IOIDataset,
    per_prompt: bool = False
):
    '''
        Return average logit difference between correct and incorrect answers
    '''
    # Get logits for indirect objects
    io_logits = logits[range(logits.size(0)), ioi_dataset.word_idx['end'], ioi_dataset.io_tokenIDs]
    s_logits = logits[range(logits.size(0)), ioi_dataset.word_idx['end'], ioi_dataset.s_tokenIDs]
    # Get logits for subject
    logit_diff = io_logits - s_logits
    return logit_diff if per_prompt else logit_diff.mean()

with t.no_grad():
    clean_logits = model(clean_dataset.toks)
    corrupt_logits = model(corr_dataset.toks)
    clean_logit_diff = ave_logit_diff(clean_logits, clean_dataset).item()
    corrupt_logit_diff = ave_logit_diff(corrupt_logits, corr_dataset).item()

def ioi_metric(
    logits: Float[Tensor, "batch seq_len d_vocab"],
    corrupted_logit_diff: float = corrupt_logit_diff,
    clean_logit_diff: float = clean_logit_diff,
    ioi_dataset: IOIDataset = clean_dataset
 ):
    patched_logit_diff = ave_logit_diff(logits, ioi_dataset)
    return (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)

def negative_ioi_metric(logits: Float[Tensor, "batch seq_len d_vocab"]):
    return -ioi_metric(logits)

def plausibility_metric(logits, ioi_dataset = clean_dataset, clean_logits = clean_logits, corrupt_logits = corrupt_logits):
    last_logits = logits[range(logits.size(0)), ioi_dataset.word_idx['end']]
    last_clean_logits = logits[range(clean_logits.size(0)), ioi_dataset.word_idx['end']]

    criterion = torch.nn.KLDivLoss()
    return criterion(last_logits, last_clean_logits)
    # - criterion(logits, corrupt_logits)

# def plausibility_metric(logits):
#     return plausibility_metric_full(logits)
    
# Get clean and corrupt logit differences
# with t.no_grad():
#     clean_metric = ioi_metric(clean_logits, corrupt_logit_diff, clean_logit_diff, clean_dataset)
#     corrupt_metric = ioi_metric(corrupt_logits, corrupt_logit_diff, clean_logit_diff, corr_dataset)

# print(f'Clean direction: {clean_logit_diff}, Corrupt direction: {corrupt_logit_diff}')
# print(f'Clean metric: {clean_metric}, Corrupt metric: {corrupt_metric}')

with t.no_grad():
    clean_metric = ioi_metric(clean_logits, corrupt_logit_diff, clean_logit_diff, clean_dataset)
    corrupt_metric = ioi_metric(corrupt_logits, corrupt_logit_diff, clean_logit_diff, corr_dataset)

print(f'Clean direction: {clean_logit_diff}, Corrupt direction: {corrupt_logit_diff}')
print(f'Clean metric: {clean_metric}, Corrupt metric: {corrupt_metric}')


# %% [markdown]
# # Run Experiment

# %%
# get the 2 fwd and 1 bwd caches; cache "normalized" and "result" of attn layers
clean_cache, corrupted_cache, clean_grad_cache = get_3_caches(
    model, 
    clean_dataset.toks,
    corr_dataset.toks,
    metric=plausibility_metric,
    mode = "edge",
)

# %%
clean_head_act = split_layers_and_heads(clean_cache.stack_head_results(), model=model)
corr_head_act = split_layers_and_heads(corrupted_cache.stack_head_results(), model=model)

# %%
stacked_grad_act = torch.zeros(
    3, # QKV
    model.cfg.n_layers,
    model.cfg.n_heads,
    clean_head_act.shape[-3], # Batch
    clean_head_act.shape[-2], # Seq
    clean_head_act.shape[-1], # D
)

for letter_idx, letter in enumerate("qkv"):
    for layer_idx in range(model.cfg.n_layers):
        stacked_grad_act[letter_idx, layer_idx] = einops.rearrange(clean_grad_cache[f"blocks.{layer_idx}.hook_{letter}_input"], "batch seq n_heads d -> n_heads batch seq d")

# %%
results = {}

for upstream_layer_idx in range(model.cfg.n_layers):
    for upstream_head_idx in range(model.cfg.n_heads):
        for downstream_letter_idx, downstream_letter in enumerate("qkv"):
            for downstream_layer_idx in range(upstream_layer_idx+1, model.cfg.n_layers):
                for downstream_head_idx in range(model.cfg.n_heads):
                    results[
                        (
                            upstream_layer_idx,
                            upstream_head_idx,
                            downstream_letter,
                            downstream_layer_idx,
                            downstream_head_idx,
                        )
                    ] = (stacked_grad_act[downstream_letter_idx, downstream_layer_idx, downstream_head_idx].cpu() * (clean_head_act[upstream_layer_idx, upstream_head_idx] - corr_head_act[upstream_layer_idx, upstream_head_idx]).cpu()).sum()

# %%
sorted_results = sorted(results.items(), key=lambda x: x[1].abs(), reverse=True)

# %%
print("Top 10 most important edges:")
for i in range(100):
    print(
        f"{sorted_results[i][0][0]}:{sorted_results[i][0][1]} -> {sorted_results[i][0][3]}:{sorted_results[i][0][4]}",
    )



# %%
