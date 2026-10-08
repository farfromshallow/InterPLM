"""
This step would expand the eval to domain-level F1, which is a more biologically meaningful metric for evaluating feature-concept alignment.
The domain-level F1 metric is described in the InterPLM paper, and is implemented in interplm/analysis/concepts/domain_level_f1.py

SAE feature activation
        ↓ threshold
Which residue is activated by the feature
        ↓
Compare with instance IDs of Region_Disordered
        ↓
How many independent disordered regions is the feature able to detect
        ↓
recall_per_domain
        ↓
precision + recall_per_domain
        ↓
F1_per_domain
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse


# 1. Locate data

DATA_ROOT = Path(
    "data/reproduction/annotations/uniprotkb/processed"
)
SHARD_DIR = DATA_ROOT / "shard_0"

CONCEPT = "Region_Disordered"
THRESHOLDS = [0.0, 0.15, 0.5, 0.6, 0.8]

FEATURE_CHUNK_SIZE = 256


# 2. Load Region_Disordered annotations

concept_names = (
    DATA_ROOT / "uniprotkb_aa_concepts_columns.txt"
).read_text().splitlines()

concept_idx = concept_names.index(CONCEPT)

concept_matrix = sparse.load_npz(
    SHARD_DIR / "aa_concepts.npz"
)

instance_ids = (
    concept_matrix[:, concept_idx]
    .toarray()
    .ravel()
)

concept_labels = instance_ids > 0

n_positive_residues = int(
    concept_labels.sum()
)

domain_ids = np.unique(
    instance_ids[instance_ids > 0]
)

n_domains = len(domain_ids)

print(f"Concept: {CONCEPT}")
print(
    f"Positive residues: "
    f"{n_positive_residues}"
)
print(
    f"Annotated region instances: "
    f"{n_domains}"
)


# 3. Load SAE activations from step 06

sae_activations = torch.load(
    SHARD_DIR / "sae_activations_layer4.pt",
    map_location="cpu",
    weights_only=True,
)

assert (
    sae_activations.shape[0]
    == len(concept_labels)
)

labels = torch.from_numpy(
    concept_labels
)

results = []


# 4. Evaluate every SAE feature

for start in range(
    0,
    sae_activations.shape[1],
    FEATURE_CHUNK_SIZE,
):

    stop = min(
        start + FEATURE_CHUNK_SIZE,
        sae_activations.shape[1],
    )

    activation_chunk = (
        sae_activations[:, start:stop]
    )

    for threshold in THRESHOLDS:

        predictions = (
            activation_chunk > threshold
        )

        true_positive_mask = (
            predictions & labels[:, None]
        )

        # Residue-level TP
        tp = (
            true_positive_mask
            .sum(dim=0)
            .numpy()
        )

        predicted_positive = (
            predictions
            .sum(dim=0)
            .numpy()
        )

        fp = predicted_positive - tp

        # Residue-level precision
        precision = np.divide(
            tp,
            tp + fp,
            out=np.zeros_like(
                tp,
                dtype=float,
            ),
            where=(tp + fp) > 0,
        )

        # Residue-level recall
        residue_recall = (
            tp / n_positive_residues
        )


# 5. Count domain-instance hits

        tp_per_domain = np.zeros(
            stop - start,
            dtype=int,
        )

        for domain_id in domain_ids:

            domain_mask = (
                torch.from_numpy(
                    instance_ids == domain_id
                )
            )

            domain_hit = (
                true_positive_mask[
                    domain_mask
                ]
                .any(dim=0)
                .numpy()
            )

            tp_per_domain += (
                domain_hit.astype(int)
            )


# 6. Domain-level recall

        recall_per_domain = (
            tp_per_domain / n_domains
        )


# 7. InterPLM-style F1_per_domain

        f1_per_domain = np.divide(
            2
            * precision
            * recall_per_domain,

            precision
            + recall_per_domain,

            out=np.zeros_like(
                precision
            ),

            where=(
                precision
                + recall_per_domain
            ) > 0,
        )


# 8. Store results

        for j, feature_id in enumerate(
            range(start, stop)
        ):

            results.append({
                "feature": feature_id,
                "threshold": threshold,

                "tp_residues":
                    int(tp[j]),

                "fp_residues":
                    int(fp[j]),

                "precision":
                    float(precision[j]),

                "residue_recall":
                    float(
                        residue_recall[j]
                    ),

                "tp_per_domain":
                    int(
                        tp_per_domain[j]
                    ),

                "n_domains":
                    n_domains,

                "recall_per_domain":
                    float(
                        recall_per_domain[j]
                    ),

                "f1_per_domain":
                    float(
                        f1_per_domain[j]
                    ),
            })


# 9. Rank results

results_df = pd.DataFrame(
    results
)

results_df = (
    results_df.sort_values(
        [
            "f1_per_domain",
            "precision",
        ],
        ascending=False,
    )
)


print(
    "\nTop 10 feature-threshold "
    "combinations by F1_per_domain:"
)

print(
    results_df.head(10).to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)


# 10. Save

output_path = (
    SHARD_DIR
    / "region_disordered_domain_f1.csv"
)

results_df.to_csv(
    output_path,
    index=False,
)

print(
    f"\nSaved results to: "
    f"{output_path}"
)

"""
How to interpret the results:
Concept: Region_Disordered
Positive residues: 1905
Annotated region instances: 35

Top 10 feature-threshold combinations by F1_per_domain:
 feature  threshold  tp_residues  fp_residues  precision  residue_recall  tp_per_domain  n_domains  recall_per_domain  f1_per_domain
     616      0.150          290          140      0.674           0.152             27         35              0.771          0.720
    6965      0.150          217          113      0.658           0.114             27         35              0.771          0.710
    1828      0.150          407          222      0.647           0.214             26         35              0.743          0.692
    5004      0.150          312          205      0.603           0.164             26         35              0.743          0.666
    5909      0.150          251           72      0.777           0.132             20         35              0.571          0.659
    4608      0.150          654          653      0.500           0.343             33         35              0.943          0.654
    7537      0.150          291          296      0.496           0.153             30         35              0.857          0.628
    4721      0.000          729          831      0.467           0.383             33         35              0.943          0.625
     746      0.000          340          253      0.573           0.178             24         35              0.686          0.625
    2802      0.150          375          462      0.448           0.197             31         35              0.886          0.595

    
Take feature 616 for example: the 290 residues distributed across 27 independent disordered regions out of 35.
This indicates that feature 616 does not cover all disordered regions, but it does appear to various disordered region instances, which meets the characteristics of a sparse feature. 
It is a more biologically meaningful metric than residue-level F1.

The differences between residue-level F1 and domain-level F1:
Residue-level F1
    ↓
Whether this can cover all residues of Region_Disordered?

Domain-level evaluation
    ↓
Does this feature appear consistently in different Region_Disordered instances?
"""