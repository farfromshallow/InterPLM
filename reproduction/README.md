# InterPLM Reproduction

This directory documents my reproduction of selected experiments from InterPLM, with the goal of understanding sparse autoencoder-based interpretability for protein language models.

## Environment

Platform: macOS, Apple Silicon (arm64)

The original `env.yml` could not be resolved on `osx-arm64` because the pinned Conda packages `pytorch::pytorch==2.8.0` and `pytorch::torchvision==0.21.0` were unavailable for this platform.

I therefore created a Python 3.11 Conda environment and installed an ARM-compatible PyTorch build via pip.

- Python: 3.11
- PyTorch: 2.14.1
- MPS available: True

## Reproduction Progress

### Level 1 — Pretrained SAE
- [ ] Load ESM-2
- [ ] Extract hidden representations
- [ ] Load a pretrained InterPLM SAE
- [ ] Inspect sparse feature activations

### Level 2 — SAE Training
- [ ] Prepare protein representations
- [ ] Train an SAE
- [ ] Evaluate reconstruction and sparsity
- [ ] Inspect learned features


## Reproduction Notes

### 1. Curated UniProtKB subset in the walkthrough is not available in the repository

The InterPLM README recommends using a curated subset of 1,000 densely annotated Swiss-Prot proteins for the concept-analysis walkthrough:

```bash
data/uniprotkb/swissprot_dense_annot_1k_subset.tsv.gz
```

However, this file is not present in the cloned repository. The repository's `.gitignore` contains a `*.gz` rule, and `git check-ignore` confirms that this path is ignored. I therefore used the README's **Option B** instead and downloaded a custom UniProtKB dataset using the provided UniProt query.

The downloaded file contains 921 reviewed mouse proteins, filtered for high-quality annotations and sequence length according to the walkthrough query.

### 2. Compatibility fix in `extract_annotations.py`

Running the annotation extraction command on the current environment initially failed during sharding with:

```text
AttributeError: 'numpy.ndarray' object has no attribute 'reset_index'
```

The original `shard_protein_data()` implementation passes the pandas DataFrame directly to `np.array_split` and then assumes that each returned shard is still a DataFrame:

```python
shards = np.array_split(df, n_shards)

for shard_id, df_shard in enumerate(shards):
    df_shard = df_shard.reset_index(drop=True)
```

In the current NumPy/pandas environment, the resulting shards are NumPy arrays, so `reset_index()` is unavailable.

I made a minimal compatibility change that splits row indices instead and then selects the corresponding rows with pandas `.iloc`:

```python
shard_indices = np.array_split(np.arange(len(df)), n_shards)

for shard_id, indices in enumerate(shard_indices):
    df_shard = df.iloc[indices].reset_index(drop=True)
```

This preserves the original intent of dividing the already shuffled DataFrame into roughly equal shards while ensuring that each shard remains a pandas DataFrame. No annotation-processing logic was changed.