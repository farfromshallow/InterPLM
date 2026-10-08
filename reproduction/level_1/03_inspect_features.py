import torch
import esm

from interplm.sae.inference import (
    load_sae_from_hf,
    get_sae_feats_in_batches,
)


# ------------------------------------------------------------
# Reproduce the ESM → SAE pipeline
# ------------------------------------------------------------

model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
model.eval()

batch_converter = alphabet.get_batch_converter()

sequence = "MKTIIALSYIFCLVFADYKDDDDK"

data = [
    ("example_protein", sequence)
]

_, _, batch_tokens = batch_converter(data)

with torch.no_grad():
    results = model(
        batch_tokens,
        repr_layers=[4],
        return_contacts=False,
    )

hidden_states = results["representations"][4]

# Remove BOS and EOS.
aa_embeddings = hidden_states[0, 1 : len(sequence) + 1]

# Load the pretrained SAE and obtain feature activations


sae = load_sae_from_hf(
    plm_model="esm2-8m",
    plm_layer=4,
)

device = next(sae.parameters()).device

with torch.no_grad():
    feature_activations = get_sae_feats_in_batches(
        sae=sae,
        device=str(device),
        aa_embds=aa_embeddings,
        chunk_size=128,
    )

print("Feature activation shape:")
print(feature_activations.shape)


strongest_feature = torch.argmax(
    feature_activations.max(dim=0).values # maximum [24 residues, 10240 features] through residues dimension
                                          #  for each feature, find the max activation across the sequence (24 residues)
).item()
# the highest-activating SAE feature across the protein sequence

feature_across_sequence = feature_activations[:, strongest_feature] 

print(
    f"\nStrongest feature in this sequence: "
    f"{strongest_feature}"
)

print("Activation across residues:")

for position, (aa, activation) in enumerate(
    zip(sequence, feature_across_sequence),
    start=1,
):
    print(
        f"{position:2d}  {aa}  "
        f"{activation.item():.4f}"
    )


# ------------------------------------------------------------
# Inspect the strongest features at each residue
# ------------------------------------------------------------

top_k = 5

print("\nTop SAE features per residue:")

for position, aa in enumerate(sequence):
    residue_activations = feature_activations[position]

    top_values, top_indices = torch.topk(
        residue_activations,
        k=top_k,
    )

    top_features = list(
        zip(
            top_indices.tolist(),
            top_values.tolist(),
        )
    )

    print(
        f"{position + 1:2d}  {aa}  "    # For position 1, amino acid M, the top 5 features are: [(feature_index_1, activation_value_1), (feature_index_2, activation_value_2), ...]
        f"{top_features}"
    )
        

# ------------------------------------------------------------
# Inspect where one feature activates across the sequence (where does the strongest feature activate across the sequence)
# ------------------------------------------------------------

"""
SAE features are not random across residues. Repeated amino-acid identities often produce recurring high-activation features.
Analysing the strongest SAE feature across a protein sequence can reveal which residues are most strongly associated with that feature, and how the feature's activation varies along the sequence.
For this example protein sequence, feature 7309 owns the highest single activation value. 
Taking feature 7309 and plotting its activation across the sequence

 1 M   0.8038   ████████████████
 2 K   0.3708   ███████
 3 T   0.1696   ███
 4 I   0.0566   █
 5 I   0.1020   ██
 6 A   0.0036
 7 L   0.0128
 8 S   0
 ...
19 K   0
20 D   0.0060
...
24 K   0.0101
So we can generate hypothesis about feature mearning:
We could come up with a hypothesis that feature 7309 might be related to the N-terminal / the beginning of the sequence.
However, we can not conclude feature 7309 represents the N-terminal of a protein sequence at this point.

Observation

    ↓

7309 activates strongly near sequence beginning

Hypothesis

    ↓

7309 may encode an N-terminal-related feature

Validation

    ↓

Does this happen across many proteins?

Does it depend on position or amino-acid identity?

Does changing the sequence alter the activation?

Does it align with known biological annotations?

Interpretation

    ↓

Only after sufficient evidence can we attach

a biological meaning to feature 7309.
"""

"""
Step 3 moves from inspecting SAE sparsity to inspecting individual feature activation patterns. 
The activation matrix can be examined from two directions. 
A residue-centric view identifies the strongest features at each residue and can reveal recurring features or candidate features worth investigating. 
A feature-centric view tracks one feature across residue positions. 
For example, feature 7309 has the highest single activation in this sequence and activates primarily near the beginning of the protein. 
This observation allows us to generate the hypothesis that feature 7309 may be associated with an N-terminal context, but validating such an interpretation requires examining the feature across many proteins and biological annotations.
"""