## What do the components of transformer do in vision transformer?

The goal of this repo is to find the analysis of various components of ViT (e.g. cls token, pos embedding..)

### Set training environment

```
- batch : 128
- lr : 1e-3
- epoch : 50
- optimizer : adam
- betas : (0.9, 0.999)
- weight_decay : 5e-5
- lr scheduler : cosine scheduler
- loss : cross entropy
- model : Vit 
```

### model 

- ViT(Vaswani) : [paper](https://arxiv.org/pdf/1706.03762.pdf)


### Results

##### attention 

![results](./figures/attention_maps.JPG)

##### CIFAR10

| model               | Batch size     | resolution | Top1-Acc          | Top5-Acc | Val Loss    | Params  |
|---------------------|----------------|------------|-------------------|----------|-------------|---------|
| ViT(vaswani)        | 128            | 32 x 32    | -                 | -        | -           | 2692426 | 

##### CIFAR100

| model               | Batch size     | resolution | Top1-Acc          | Top5-Acc | Val Loss    | Params  |
|---------------------|----------------|------------|-------------------|----------|-------------| ------  |
| ViT(vaswani)        | 128            | 32 x 32    | 0.7044            | -        | -           |         |


#### Imagenet1K

| Model          | Batch size     | Resolution | Top1-Acc          | Top5-Acc | Val Loss | Params            |
|----------------|----------------|------------|-------------------|----------|----------| ----------------- |
| ViT(B)         | 256            | 224 x 224  | 77.31             | -        | -        | 86540008 (86.5M)  |
