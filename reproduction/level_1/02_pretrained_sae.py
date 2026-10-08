"""
This step demonstrates how to pass the ESM-2 representations through InterPLM's pretrained SAE 
to obtain a sparse representation for each amino acid in a protein sequence.

320 dimensional ESM representations are encoded by the SAE into sparse representations
"""
import torch
import esm

from interplm.sae.inference import (
    load_sae_from_hf,
    get_sae_feats_in_batches,
)


# 1. Load the same ESM-2 model as step 1 in 01_esm_representation.py
model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
model.eval()

batch_converter = alphabet.get_batch_converter()

sequence = "MKTIIALSYIFCLVFADYKDDDDK"

data = [
    ("example_protein", sequence)
]

_, _, batch_tokens = batch_converter(data)


# 2. Extract layer-4 ESM representations
with torch.no_grad():
    results = model(
        batch_tokens,
        repr_layers=[4],
        return_contacts=False,
    )

hidden_states = results["representations"][4]

# Step 2 add-ons
print("Raw ESM hidden-state shape:")
print(hidden_states.shape)


# 3. Remove ESM special tokens
aa_embeddings = hidden_states[0, 1 : len(sequence) + 1]

print("\nAmino-acid embedding shape:")
print(aa_embeddings.shape)


# 4. Load InterPLM's pretrained SAE
sae = load_sae_from_hf(
    plm_model="esm2-8m",
    plm_layer=4,
)
# pre-defined SAE dictionary size is 10240, x ∈ R^320
# learned directions / basis-like feature vectors
print("\nSAE dictionary size:")
print(sae.dict_size)


# 5. Find which device the SAE is using
device = next(sae.parameters()).device

print("\nSAE device:")
print(device)


# 6. Pass the ESM representations through the SAE
with torch.no_grad():
    feature_activations = get_sae_feats_in_batches(
        sae=sae,
        device=str(device),
        aa_embds=aa_embeddings,
        chunk_size=128,
    )

print("\nSAE feature activation shape:")
print(feature_activations.shape)
# [num of amino acids, SAE dictionary size]

# 7. Count active features for each amino acid
active_features_per_residue = (feature_activations > 0).sum(dim=1)

print("\nActive SAE features per residue:")
for position, (aa, count) in enumerate(
    zip(sequence, active_features_per_residue),
    start=1,
):
    print(
        f"{position:2d}  {aa}  "
        f"{count.item()} active features"
    )

"""    
The raw ESM hidden state has shape [1, 26, 320], corresponding to one protein, 24 amino-acid tokens plus two special tokens, and a 320-dimensional representation for each token. 
After removing the special tokens, the amino-acid representations have shape [24, 320]. 
The pretrained SAE maps each 320-dimensional contextual representation into a sparse feature space containing 10,240 learned features, so the SAE activation tensor has shape [24, 10240]. 
For each residue, only a small subset of these 10,240 features has a non-zero activation.
"""