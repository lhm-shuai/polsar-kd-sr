"""Reference implementation for

    Feedback Learning-Based Polarimetric Scattering Representation for
    PolSAR Ship Detection

The pipeline follows the paper's stages:

    Stage I    matlab/decompose_components.m, matlab/make_hard_labels.m
    Stage II   src/teacher.py
    Stage III  src/sr.py
    evaluation src/evaluate.py, src/detector.py
"""
__version__ = "1.0.0"
