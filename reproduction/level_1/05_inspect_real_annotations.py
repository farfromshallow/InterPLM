"""
expected data structure of the processed annotation data:
                     aa_concepts.npz

                   concept 0   concept 1   concept 2 ...
                         │          │          │
metadata row 0   M       0          0          1
metadata row 1   K       0          1          1
metadata row 2   T       0          1          0
...
"""

from pathlib import Path
import pandas as pd
from scipy import sparse


# 1. Locate processed annotation data

data_root = Path(
    "data/reproduction/annotations/uniprotkb/processed"
)

shard_id = 0
shard_dir = data_root / f"shard_{shard_id}"

metadata_path = shard_dir / "aa_metadata.csv"
concept_matrix_path = shard_dir / "aa_concepts.npz"
concept_names_path = data_root / "uniprotkb_aa_concepts_columns.txt"


# 2. Load residue metadata

metadata = pd.read_csv(metadata_path)

print("Metadata shape:")
print(metadata.shape)

print("\nFirst residues:")
print(metadata.head())


# 3. Load concept names

with open(concept_names_path) as f:
    concept_names = [
        line.strip()
        for line in f
        if line.strip()
    ]

print("\nNumber of concepts:")
print(len(concept_names))

print("\nFirst 20 concepts:")
for i, concept in enumerate(concept_names[:20]):
    print(i, concept)


# 4. Load sparse concept matrix

concept_matrix = sparse.load_npz(
    concept_matrix_path
)

print("\nConcept matrix shape:")
print(concept_matrix.shape)


# 5. Check alignment

assert concept_matrix.shape[0] == len(metadata)
assert concept_matrix.shape[1] == len(concept_names)

print("\nMatrix alignment checks passed.")

"""
There are 28754 residues x 139 concepts in the first shard of the processed annotation data.
concept_matrix [i, j] can be explained as 'the i-th residue is annotated with the j-th concept'.
"""

# 6. Count positive residues for each concept in shard 0. 
# This is useful for inspecting concept prevalence and selecting
# concepts with sufficient support for the following analysis.
positive_counts = (
    (concept_matrix > 0)
    .sum(axis=0)
    .A1
)

concept_counts = list(
    zip(concept_names, positive_counts)
)

concept_counts.sort(
    key=lambda x: x[1],
    reverse=True,
)

print("\nTop 20 concepts by number of positive residues:")

for concept, count in concept_counts[:20]:
    print(f"{concept:<50} {count}")


# 7. Inspect biological concepts present in this shard

excluded_prefix = "amino_acid_" 
# exclude concepts that are just amino acid types, since they are less helpful for understanding protein bological / structural annotations.

present_biological_concepts = [
    (i, concept_names[i], positive_counts[i])
    for i in range(len(concept_names))
    if positive_counts[i] > 0
    and not concept_names[i].startswith(excluded_prefix)
]

present_biological_concepts.sort(
    key=lambda x: x[2],
    reverse=True
)

print("\nBiological concepts present in shard 0:")

for concept_idx, concept, count in present_biological_concepts:
    print(
        f"{concept_idx:3d} | "
        f"{concept:<50} | "
        f"{count}"
    )

# 8. Extract one real biological concept as a binary residue-level label vector

"""
Region_Disordered is frequent enough in shard 0 to inspect and is more biologically informative than amino-acid identity or the broad "*_any" labels.
UniProt annotation
       ↓
extract_annotations.py
       ↓
aa_concepts.npz
       ↓
column 91: Region_Disordered
       ↓
[0, 0, 0, 1, 1, 1, ...]
       ↑
28754 residues

1095 means 1905 residues that are annotated as disordered region in this shard.

"""
selected_concept = "Region_Disordered"
selected_concept_idx = concept_names.index(selected_concept)

# The sparse matrix may contain positive instance IDs rather than only 0/1.
# For residue-level concept presence, any value > 0 is treated as positive.
concept_labels = (
    concept_matrix[:, selected_concept_idx] > 0
).astype(int).toarray().ravel()

print(f"\nSelected concept: {selected_concept}")
print(f"Concept column: {selected_concept_idx}")
print(f"Positive residues: {concept_labels.sum()}")
print(f"Total residues: {len(concept_labels)}")


# Inspect the metadata rows corresponding to positive annotations.

positive_indices = concept_labels.nonzero()[0]

annotated_residues = metadata.iloc[
    positive_indices
].copy()

print("\nFirst 20 annotated residues:")
print(annotated_residues.head(20))