# Causality In Traffic Accident
Repository for Traffic Accident Benchmark for Causality Recognition (ECCV 2020)

## Overview
<img width="480px" src="overview.png">

## Data Preparation

## Cause and Effect Event Classification
We implement Temporal Segment Networks (ECCV 2016)
- The default argument for code is TSN with average consensus function.
```
python train_classifier.py
```
- By passing argument below, TSN with linear consensus function can be tested.
```
python train_classifier.py --consensus_function linear
```

## Temporal Cause and Effect Event Localization
