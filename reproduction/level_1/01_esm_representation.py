"""
This step demonstrates how to use a pretrained ESM-2 model to obtain representations for a protein sequence. 
It loads the ESM-2 8M model, processes a short example protein sequence, and retrieves the hidden states from layer 4 of the model.
batch_tokens (through ESM-2 8M)-> hidden_states [1, 26, 320]
"""
import torch
import esm


# Load ESM-2 8M
model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
model.eval()

batch_converter = alphabet.get_batch_converter()

# A short example protein sequence
# sequence length is 24 amino acids, while esm adds 2 special tokens, so the total length is 26
data = [
    ("example_protein", "MKTIIALSYIFCLVFADYKDDDDK")
]

batch_labels, batch_strs, batch_tokens = batch_converter(data)

print("Input sequence:")
print(batch_strs[0])

print("\nToken tensor shape:")
print(batch_tokens.shape)


# Run ESM-2 and request the representation from layer 4
with torch.no_grad():
    results = model(
        batch_tokens,
        repr_layers=[4],
        return_contacts=False,
    )

hidden_states = results["representations"][4]

print("\nLayer 4 hidden-state shape:")
print(hidden_states.shape)

print("\nFirst token representation:")
print(hidden_states[0, 0, :10])

print("\nFirst amino-acid representation:")
print(hidden_states[0, 1, :10])