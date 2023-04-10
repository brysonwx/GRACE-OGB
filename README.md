# GRACE-OGB
Support OGB datasets in [GRACE](https://github.com/CRIPAC-DIG/GRACE).

## Dependencies

```shell
<username>@node02:~/projects/GRACE$ pip list | grep torch
torch               1.12.0+cu116
torch-cluster       1.6.1
torch-geometric     2.0.4
torch-scatter       2.1.1
torch-sparse        0.6.17
torch-spline-conv   1.2.2
torchaudio          0.12.0+cu116
torchvision         0.13.0+cu116
```

Install other necessary dependencies with
```
pip install -r requirements.txt
```

## Usage

Train and evaluate the model by executing
```
python train.py --gpu_id 2 --dataset ogbn-arxiv
```
The `--dataset` argument should be one of [ Cora, CiteSeer, PubMed, DBLP, ogbn-arxiv, ogbn-products].