# alpha-leak-data-simulator

1. Overview
The framework will use WNTR to simulate water networks, add various leak scenarios, and extract features for a graph-based deep learning approach to leak detection. The goal is to create a pipeline that:

Simulates water networks under stationary nighttime conditions
Introduces leaks at different locations and with varying severities
Extracts comprehensive node, edge, and network-level features
Creates data augmentation through subgraph extraction and permutation

2. Architecture
epanet_simulator/
├── __init__.py
├── simulator.py          # Core simulation functionality
├── feature_extractor.py  # Extract features from network
├── leak_generator.py     # Generate leak scenarios
├── subgraph_manager.py   # Handle subgraph extraction
├── data_augmentor.py     # Handle data permutation and augmentation
├── utils/
│   ├── __init__.py
│   ├── io.py             # Input/output operations
│   ├── visualization.py  # Network visualization tools
│   └── matrix.py         # Matrix operations for network features
└── dataset/
    ├── __init__.py
    └── dataset_generator.py  # Generate complete datasets for ML