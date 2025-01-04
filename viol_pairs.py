import numpy as np

# Load the adjusted labels (if not already loaded)
adjusted_labels = np.load("adjusted_labels.npy")

# Load the must-link and cannot-link pairs (if not already loaded)
must_link_pairs = np.load("must_link_pairs.npy", allow_pickle=True).tolist()
cannot_link_pairs = np.load("cannot_link_pairs.npy", allow_pickle=True).tolist()

# --- Check for cannot-link violations ---
print("\nChecking for cannot-link violations:")
cannot_link_violations = 0
for i, j in cannot_link_pairs:
    if adjusted_labels[i] != -1 and adjusted_labels[j] != -1 and adjusted_labels[i] == adjusted_labels[j]:
        print(f"Violation: Pair ({i}, {j}) in the same cluster ({adjusted_labels[i]})")
        cannot_link_violations += 1

if cannot_link_violations == 0:
    print("No cannot-link constraint violations found!")
else:
    print(f"Total cannot-link constraint violations: {cannot_link_violations}")
# ----------------------------------------

# --- Check for must-link violations ---
print("\nChecking for must-link violations:")
must_link_violations = 0
for i, j in must_link_pairs:
    if adjusted_labels[i] != -1 and adjusted_labels[j] != -1 and adjusted_labels[i] != adjusted_labels[j]:
        print(f"Violation: Pair ({i}, {j}) in different clusters ({adjusted_labels[i]}, {adjusted_labels[j]})")
        must_link_violations += 1

if must_link_violations == 0:
    print("No must-link constraint violations found!")
else:
    print(f"Total must-link constraint violations: {must_link_violations}")
# ----------------------------------------