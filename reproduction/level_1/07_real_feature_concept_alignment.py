"""
Evaluate pretrained SAE features against Region_Disordered.

Single-shard exploratory reproduction of InterPLM's
residue-level feature-concept evaluation.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse



# 1. Configuration

data_root = Path(
    "data/reproduction/annotations/uniprotkb/processed"
)

shard_dir = data_root / "shard_0"

selected_concept = "Region_Disordered"

thresholds = [0.0, 0.15, 0.5, 0.6, 0.8]

feature_chunk_size = 256



# 2. Load real concept annotations

concept_names = (
    data_root / "uniprotkb_aa_concepts_columns.txt"
).read_text().splitlines()

concept_idx = concept_names.index(selected_concept)

concept_matrix = sparse.load_npz(
    shard_dir / "aa_concepts.npz"
)

instance_ids = (
    concept_matrix[:, concept_idx]
    .toarray()
    .ravel()
)

concept_labels = instance_ids > 0

n_positive = int(concept_labels.sum())

print(f"Concept: {selected_concept}")
print(f"Concept column: {concept_idx}")
print(f"Positive residues: {n_positive}")



# 3. Load pretrained SAE activations

sae_activations = torch.load(
    shard_dir / "sae_activations_layer4.pt",
    map_location="cpu",
    weights_only=True,
)

assert sae_activations.shape[0] == len(concept_labels)

n_residues, n_features = sae_activations.shape

print(f"Residues: {n_residues}")
print(f"SAE features: {n_features}")



# 4. Compare all features with the selected concept

labels = torch.from_numpy(concept_labels)

results = []

for start in range(0, n_features, feature_chunk_size):

    stop = min(start + feature_chunk_size, n_features)

    activation_chunk = sae_activations[:, start:stop]

    for threshold in thresholds:

        predictions = activation_chunk > threshold

        tp = (
            predictions & labels[:, None]
        ).sum(dim=0).numpy()

        predicted_positive = (
            predictions.sum(dim=0).numpy()
        )

        fp = predicted_positive - tp

        fn = n_positive - tp

        precision = np.divide(
            tp,
            tp + fp,
            out=np.zeros_like(tp, dtype=float),
            where=(tp + fp) > 0,
        )

        recall = tp / n_positive

        f1 = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros_like(precision),
            where=(precision + recall) > 0,
        )

        for j, feature_id in enumerate(range(start, stop)):

            results.append({
                "feature": feature_id,
                "threshold": threshold,
                "tp": int(tp[j]),
                "fp": int(fp[j]),
                "fn": int(fn[j]),
                "precision": float(precision[j]),
                "recall": float(recall[j]),
                "f1": float(f1[j]),
            })



# 5. Rank feature-concept associations

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    ["f1", "precision"],
    ascending=False,
)

print("\nTop 10 feature-threshold combinations:")

print(
    results_df.head(10).to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)



# 6. Save results

output_path = (
    shard_dir / "region_disordered_feature_f1.csv"
)

results_df.to_csv(
    output_path,
    index=False,
)

print(f"\nSaved results to: {output_path}")

"""
Concept: Region_Disordered
Concept column: 91
Positive residues: 1905
Residues: 28754
SAE features: 10240

Top 10 feature-threshold combinations:
 feature  threshold   tp   fp   fn  precision  recall    f1
    1828      0.000  988 1584  917      0.384   0.519 0.441
    3495      0.000  978 1568  927      0.384   0.513 0.439
    5909      0.000  763  889 1142      0.462   0.401 0.429
     616      0.000  836 1186 1069      0.413   0.439 0.426
    5004      0.000  799 1065 1106      0.429   0.419 0.424
    4721      0.000  729  831 1176      0.467   0.383 0.421
    7091      0.000 1000 1884  905      0.347   0.525 0.418
    4608      0.000 1110 2361  795      0.320   0.583 0.413
    6965      0.000  846 1368 1059      0.382   0.444 0.411
    4608      0.150  654  653 1251      0.500   0.343 0.407


How to interpret the results:
- Most of the top features have a threshold of 0.0, which means that they relies on whether this feature is active in the given context, rather than a feature is strongly activated.
"""