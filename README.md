# Causality In Traffic Accident
Repository for Traffic Accident Benchmark for Causality Recognition (ECCV 2020)

## Overview
<img width="480px" src="figures/overview.png">

Main contributions of the [paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123520528.pdf)
- We introduce a traffic accident analysis benchmark, denoted by CTA, which contains temporal intervals of a cause and an effect in each accident and their semantic labels provided by [the crash avoidance research](https://rosap.ntl.bts.gov/view/dot/6281).
- We construct the dataset based on the semantic taxonomy in the crash avoidance research, which makes the distribution of the benchmark coherent to the semantic taxonomy and the real-world statistics.
- We analyze traffic accident tasks by comparing multiple algorithms for temporal cause and effect event localization.

## Dataset Preparation
You can download the dataset in the below link
[Details of dataset](dataset/DATASET.md)

## Benchmark
### Cause and Effect Event Classification
We adopt Temporal Segment Networks (ECCV 2016) in our benchmark.
- The default arguments for code are set to train TSN with average consensus function.
```
python train_classifier.py --consensus_type average --random_seed 17
python train_classifier.py --consensus_type linear --random_seed 3
```

- The performance of classification models with above arguments is shown in below.

| TSN     | Cause Top-1 | Cause Top-2 | Effect Top-1 | Effect Top-2 |
| ------- |:-----------:|:-----------:|:------------:|:------------:|
| Average | 25.00       | 32.25       | 43.75        | 87.50        |
| Linear  | 31.25       | 37.50       | 87.50        | 93.75        |


### Temporal Cause and Effect Event Localization
We adopt three types of baseline methods (single-stage action detection, proposal-based action detection and action segmentation) in our benchmark.
Our implementation of methods is based on below three works.

SST: Single-Stream Temporal Action Proposals, CVPR 17
R-C3D: Region Convolutional 3D Network for Temporal Activity Detection, ICCV 2017
MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation, CVPR 19


- Single-stage Action Detection
```
python train_localization.py --architecture_type forward-SST
python train_localization.py --architecture_type backward-SST
python train_localization.py --architecture_type bi-SST
python train_localization.py --architecture_type SSTCN-SST --num_layers 10 --num_epochs 100
```

| SST     | Cause IoU > 0.5 | Effect IoU > 0.5 | Cause IoU > 0.7 | Effect IoU > 0.7 |
| ------- |:-----------:|:-----------:|:------------:|:------------:|
| Forward   | 9.66  | 22.41 | 5.17  | 7.24  |
| Backward  | 20.34 | 34.83 | 7.24  | 13.10 |
| Bi        | 20.69 | 33.10 | 10.34 | 14.83 | 
| SSTCN     | 25.17 | 35.52 | 10.00 | 12.41 | 

For single-stage detection, we adopt SST. We use K = 128 for the size of the hidden dimension for gated recurrent units (GRU). To change the proposed method into a single-stage detection method, we simply change the class prediction layer to have three classes background, cause and effect—and substitute binary cross-entropy loss function into cross-entropy loss function. We use 64 anchor boxes with temporal scales [1 · δ, 2 · δ, · · · , K · δ] in seconds, where δ = 0.32 seconds and K = 64.

Note that the performances of backward-SST, Bi-SST and SSTCN-SST except forward-SST are better than those in the paper.

- Action Segmentation
```
python train_localization.py --architecture_type SSTCN-Segmentation --num_layers 
python train_localization.py --architecture_type MSTCN-Segmentation
```

- Proposal-based Action Detection (not supported yet)
```
python train_localization.py --architecture_type naive-conv-R-C3D
python train_localization.py --architecture_type SSTCN-R-C3D
``` 

### Citation

```
@inproceedings{you2020CTA,
    title     = "{Traffic Accident Benchmark for Causality Recognition}",
    author    = {You, Tackgeun and Han, Bohyung},
    booktitle = {ECCV},
    year      = {2020}
}
```
