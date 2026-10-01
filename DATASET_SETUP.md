# Carinthia-S Dataset Setup

The raw Carinthia-S dataset is intentionally excluded from the final project ZIP
to keep the research package manageable.

Place the dataset under:

data/raw/carinthia_s/data/

Expected structure:

data/
└── raw/
    └── carinthia_s/
        └── data/
            ├── images/
            ├── masks/
            └── carinthia-s.csv

Current experiment verification:

- Images: 4591
- Masks: 4591
- Image dimensions: 480 x 480
- Image mode: grayscale (L)
- CSV separator: semicolon (;)
- Empty masks: 226
- Non-empty masks: 4365

Ground-truth rule:

- Empty mask = REFERENCE / NORMAL
- Non-empty mask = ANOMALY / DEFECT

After placing the dataset, run:

python experiments/01_verify_dataset.py
