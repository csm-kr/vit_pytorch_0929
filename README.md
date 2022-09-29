## What do the components of transformer do in vision transformer?

The goal of this repo is to find the analysis of various components of ViT (e.g. cls token, pos embedding..)

### Set training environment

- batch : 128
- lr : 1e-3
- epoch : 100

- optimizer : adamw
- betas : (0.9, 0.999)
- weight_decay : 5e-2

- warm up epoch : 10 
- lr scheduler : cosine scheduler
- loss : label smooth cross entropy
- model : Vit 

### model 


![results](./figures/attention_maps.JPG)

- vit(Vaswani)

### Experiments 

![results](./figures/performance_small_size.JPG)

reference : https://arxiv.org/pdf/2112.13492.pdf

#### CIFAR10

| model               | Batch size     | resolution | Top1-Acc          | Top5-Acc | Val Loss    | Params  |
|---------------------|----------------|------------|-------------------|----------|-------------|---------|
| ViT(paper)          | 128            | 32 x 32    | 0.9358            | -        | -           |         |
| ViT(Ours)           | 128            | 32 x 32    | 0.9379 (+0.21%)   | -        | 0.6577 (96) |         |
| ViT(vaswani)        | 128            | 32 x 32    | -                 | -        | -           | 2692426 | 
| TomViT              | 128            | 32 x 32    | 0.9524 (+1.66%)   | -        | 0.6265 (98) | 2679370 | 
| TomViT AE(xavier u) | 128            | 32 x 32    | 0.9496            | -        | -           | 3315949 | 
| TomViT AE2          | 128            | 32 x 32    | 0.9557 (+1.99%)   | -        | -           | 3315949 | 

#### CIFAR100

| model               | Batch size     | resolution | Top1-Acc          | Top5-Acc | Val Loss    | Params  |
|---------------------|----------------|------------|-------------------|----------|-------------| ------  |
| ViT                 | 128            | 32 x 32    | 0.7381            | -        | -           |         |
| ViT(svit)           | 128            | 32 x 32    | 0.7246 (-1.35%)   | -        | -           |         |
| ViT(Ours)           | 128            | 32 x 32    | 0.7289 (-0.92%)   | -        | 1.71078     |         |
| ViT(vaswani)        | 128            | 32 x 32    | 0.7044            | -        | -           |         |
| TomVit              | 128            | 32 x 32    | 0.7548 (+1.67%)   | -        | 1.62798     | 2696740 |
| TomVit AE(xavier u) | 128            | 32 x 32    | 0.7532            | -        | 1.682       | -       |
| TomVit AE2          | 128            | 32 x 32    | 0.7737 (+3.56%)   | -        | 1.5951      | 3333319 |

#### CIFAR10 server

2022.08.30

| epoch | model                 | Batch size     | resolution | Top1-Acc          | Top5-Acc | Val Loss    | Params  |
|---------|---------------------|----------------|------------|-------------------|----------|-------------|---------|
|         | ViT(vaswani-4AE)    | 128            | 32 x 32    | 0.8840            | -        | -           |         |
| 310     | ViT(vaswani-1AE)    | 128            | 32 x 32    | 0.9372            | -        | -           |         |
| 310     | ViT(vaswani-1AE)    | 128            | 32 x 32    | 0.9372            | -        | -           |         |


#### Imagenet1K

- from scratch model 
```
- batch : 256
- init lr : 1e-3
- epoch : 310
- optimizer : adamw
- betas : (0.9, 0.999)
- weight_decay : 5e-2
- warm up epoch : 10 
- lr scheduler : cosine scheduler (min_lr : 1e-5)
- loss : label smooth cross entropy
- model : Vit 
```
![imagenet_vit_vaswaini](./figures/imagenet_vit_vaswani.JPG)


| Model          | Batch size     | Resolution | Top1-Acc          | Top5-Acc | Val Loss | Params            |
|----------------|----------------|------------|-------------------|----------|----------| ----------------- |
| ViT(B)         | 256            | 224 x 224  | 77.31             | -        | -        | 86540008 (86.5M)  |
| TomViT(B)         | 256            | 224 x 224  | 77.816             | -        | -        | 86540008 (86.5M)  |
| ViT(T2T paper) | 512            | 224 x 224  | 79.8              | -        | -        | 86.4M  |