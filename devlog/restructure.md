- check file discovery. Recursive tif file discovery or not? (Decide).


- No training capability needed in production pipeline. Move to separate `training/` package installable via extras.
Installaition:
pip install aggrequant[training]  # For model development only

  Architecture Suggestion

  Current structure (complex):
  ├── aggrequant/
  │   ├── nn/
  │   │   ├── architectures/     # 8 UNet variants, factory pattern
  │   │   ├── training/          # Full trainer (unused)
  │   │   ├── evaluation/        # Metrics (unused)
  │   │   └── data/              # 3 dataset classes (2 unused)
  │   └── segmentation/
  │       └── cells/
  │           ├── cellpose.py           # Used
  │           └── distance_intensity.py # Unused

  Suggested structure (simple):
  ├── aggrequant/                # Core inference package
  │   ├── nn/
  │   │   └── unet.py           # Just ModularUNet class
  │   └── segmentation/
  │       └── cells/
  │           └── cellpose.py   # Single cell segmentation method
  │
  └── aggrequant-training/      # Separate optional package
      ├── architectures/        # Experiment with architectures
      ├── trainer.py
      └── losses.py

