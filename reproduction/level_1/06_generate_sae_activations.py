"""
Generate pretrained SAE feature activations for all residues in shard 0.

Pipeline:

protein_data.tsv
    ↓
protein sequences
    ↓
ESM-2 8M, layer 4
    ↓
residue embeddings [N, 320]
    ↓
pretrained InterPLM SAE
    ↓
feature activations [N, 10240]

The final activation rows must correspond to the residue rows in aa_metadata.csv and aa_concepts.npz.
"""

from pathlib import Path

import esm
import pandas as pd
import torch

from interplm.sae.inference import (
    get_sae_feats_in_batches,
    load_sae_from_hf,
)



# 1. Locate shard-0 data

data_root = Path(
    "data/reproduction/annotations/uniprotkb/processed"
)

shard_dir = data_root / "shard_0"

protein_data_path = shard_dir / "protein_data.tsv"
metadata_path = shard_dir / "aa_metadata.csv"

output_path = shard_dir / "sae_activations_layer4.pt"



# 2. Load protein data and residue metadata

protein_data = pd.read_csv(
    protein_data_path,
    sep="\t",
)

metadata = pd.read_csv(
    metadata_path,
)

protein_sequences = (
    protein_data
    .set_index("Entry")["Sequence"]
    .to_dict()
)

print(f"Proteins: {len(protein_data)}")
print(f"Residues: {len(metadata)}")


# ------------------------------------------------------------
# 3. Load ESM-2 8M
# ------------------------------------------------------------

esm_model, alphabet = (
    esm.pretrained.esm2_t6_8M_UR50D()
)

esm_model.eval()

batch_converter = alphabet.get_batch_converter()

esm_layer = 4


# ------------------------------------------------------------
# 4. Load pretrained InterPLM SAE
# ------------------------------------------------------------

sae = load_sae_from_hf(
    plm_model="esm2-8m",
    plm_layer=esm_layer,
)

sae.eval()

sae_device = str(
    next(sae.parameters()).device
)

print(f"SAE device: {sae_device}")
print(f"SAE dictionary size: {sae.dict_size}")



# 5. Generate SAE activations protein by protein

all_activations = []

protein_groups = metadata.groupby(
    "Entry",
    sort=False,
)

n_proteins = metadata["Entry"].nunique()


for protein_number, (entry, entry_metadata) in enumerate(
    protein_groups,
    start=1,
):

    sequence = protein_sequences[entry]

    # One lightweight alignment guard:
    # metadata residue order must reproduce the original sequence.
    metadata_sequence = "".join(
        entry_metadata["amino_acid"].tolist()
    )

    assert metadata_sequence == sequence

    # ESM automatically adds BOS and EOS.
    _, _, tokens = batch_converter(
        [(entry, sequence)]
    )

    with torch.no_grad():

        outputs = esm_model(
            tokens,
            repr_layers=[esm_layer],
            return_contacts=False,
        )

        embeddings = outputs[
            "representations"
        ][esm_layer]

        # Shape before removal:
        # [1, sequence_length + 2, 320]
        #
        # Remove BOS and EOS.
        residue_embeddings = embeddings[
            0,
            1 : len(sequence) + 1,
            :
        ]

        # Shape:
        # [sequence_length, 320]

        activations = get_sae_feats_in_batches(
            sae=sae,
            device=sae_device,
            aa_embds=residue_embeddings,
            chunk_size=128,
        )

    all_activations.append(
        activations.cpu()
    )

    if (
        protein_number % 10 == 0
        or protein_number == n_proteins
    ):
        print(
            f"Processed "
            f"{protein_number}/{n_proteins} proteins"
        )



# 6. Combine proteins into one residue × feature matrix

sae_activations = torch.cat(
    all_activations,
    dim=0,
)

print(
    "\nFinal SAE activation shape:",
    sae_activations.shape,
)



# 7. Final dimensional checks

assert (
    sae_activations.shape[0]
    == len(metadata)
)

assert (
    sae_activations.shape[1]
    == sae.dict_size
)

print("Activation dimensions match annotation metadata.")



# 8. Save activations

torch.save(
    sae_activations,
    output_path,
)

print(
    f"\nSaved activations to:\n{output_path}"
)