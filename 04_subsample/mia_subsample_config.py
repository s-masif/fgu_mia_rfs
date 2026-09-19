# Seven medium/large datasets with room to shrink to 25% and still train.
DATASETS = [
    "roman_empire",      # 22,662 nodes, 18 classes, heterophilous
    "amazon_ratings",    # 24,492 nodes,  5 classes, heterophilous
    "Computers",         # 13,381 nodes, 10 classes
    "WikiCS",            # 11,701 nodes, 10 classes
    "tolokers",          # 11,758 nodes,  2 classes
    "minesweeper",       # 10,000 nodes,  2 classes
    "Photo",             #  7,487 nodes,  8 classes
]

SCENARIOS = ["client", "node"]

# Subsample fractions. 1.0 is the full graph (anchors against the main table).
FRACTIONS = [1.0, 0.75, 0.50, 0.25]

SEEDS = [7, 42, 99]

# Sampler: Metropolis-Hastings random walk (degree-corrected). See subsample.py.
SAMPLER = "mhrw"

# Fixed federated / unlearning settings so that SIZE is the only axis that
# varies (topology is held by the sampler; these are held by config).
DEFAULT_PARTITIONING = "louvain"
DEFAULT_CLIENT_FORGET_R = 0.20
DEFAULT_NODE_FORGET_R   = 0.10
# num_clients uses each dataset's per-dataset default from DATASET_CONFIGS.

# Combination count:
#   7 datasets x 2 scenarios x 4 fractions x 3 seeds = 168 configurations.
