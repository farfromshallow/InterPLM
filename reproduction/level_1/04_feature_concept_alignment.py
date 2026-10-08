"""
compare_activations.py evaluates every SAE feature against every biological concept by thresholding feature activations and counting how often feature activation agrees or disagrees with concept annotations across the evaluation data.
Analysing the compare_activation.py script to get the eval logic for feature-concept alignment.
There are two metrics to evaluate the alignment between a feature and a biological concept:
1. SAE feature metrix: samples x features
2. concept annotation matrix: samples x concepts (annotation of biological concepts for each sample)
to figure out for each feature and concept pair, can the activation value of the feature predict the presence of the concept annotation?

Then binarize the SAE feature matrix according to a threshold and the concept annotation matrix to get a binary classification problem for each feature-concept pair.
This allows us to see if one feature is active enough to predict the presence of a biological concept, and if so, how well it does so.

For feature f
and biological concept c:

How often does f activate
when c is actually annotated?

How often does f activate
when c is NOT annotated?

the role of settig thresholds for feature activation: 
activation threshold
        ↓
precision / recall trade-off
        ↓
F1

Step 3 tries to find the qualitative relationship between SAE features and biological concepts.
Step 4 is to quantify the relationship between SAE features and biological concepts.

With scripts in interplm/analysis/concepts, we can draw the whole pipeline of feature-concept alignment:
                 SAE training
                     │
                     ↓
              learned features
                     │
                     ↓
        ┌─────────────────────────┐
        │ biological annotations │
        └─────────────────────────┘
                     │
                     ↓
          FEATURE × CONCEPT
                     │
              threshold activation
                     │
                     ↓
               TP / FP / FN
                     │
                     ↓
        F1 + domain-level F1
                     │
                     ↓
              VALIDATION SET
                     │
          select feature–concept
              associations
                     │
                     ↓
                TEST SET
                     │
        evaluate selected features
                     │
                     ↓
       interpretable associations
       that generalize to new data
"""


from interplm.analysis.concepts.calculate_f1 import calculate_metrics
import numpy as np


# ------------------------------------------------------------
# Toy feature activations and concept annotations
# ------------------------------------------------------------

# Activation of one SAE feature across 10 residue positions.
feature_activations = np.array([
    0.91,
    0.82,
    0.76,
    0.63,
    0.51,
    0.42,
    0.31,
    0.18,
    0.07,
    0.00,
])

# Gold biological annotation for the same 10 residues.
# 1 = concept is present
# 0 = concept is absent
concept_labels = np.array([
    1,
    1,
    1,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
])



# Binarize feature activations using a threshold

threshold = 0.5

feature_predictions = (
    feature_activations > threshold
).astype(int)

print("Feature activations:")
print(feature_activations)

print("\nConcept labels:")
print(concept_labels)

print(f"\nFeature predictions at threshold {threshold}:")
print(feature_predictions)

# Calculate TP, FP, and FN

metrics = calculate_metrics(
    feature_activations,
    concept_labels,
    threshold,
)

feature_predictions = (
    feature_activations > threshold
).astype(int)

print("\nConfusion counts:")
print(f"TP: {metrics['tp']}")
print(f"FP: {metrics['fp']}")
print(f"FN: {metrics['fn']}")
print("\nMetrics:")
print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall:    {metrics['recall']:.3f}")
print(f"F1:        {metrics['f1']:.3f}")

# ------------------------------------------------------------
# Evaluate the same feature at multiple thresholds
# ------------------------------------------------------------

thresholds = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8]

results = []

print("\nThreshold comparison:")

for threshold in thresholds:

    predictions = (
        feature_activations > threshold
    ).astype(int)

    tp = np.sum(
        (predictions == 1)
        & (concept_labels == 1)
    )

    fp = np.sum(
        (predictions == 1)
        & (concept_labels == 0)
    )

    fn = np.sum(
        (predictions == 0)
        & (concept_labels == 1)
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    results.append({
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    })

    print(
        f"threshold={threshold:.2f} | "
        f"precision={precision:.3f} | "
        f"recall={recall:.3f} | "
        f"F1={f1:.3f}"
    )


# ------------------------------------------------------------
# Find the threshold with the highest F1
# ------------------------------------------------------------

best_result = max(
    results,
    key=lambda result: result["f1"]
)

# Find the best threshold based on the highest F1 score
print("\nBest threshold:")
print(
    f"threshold={best_result['threshold']:.2f} | "
    f"precision={best_result['precision']:.3f} | "
    f"recall={best_result['recall']:.3f} | "
    f"F1={best_result['f1']:.3f}"
)