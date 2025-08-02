# MoCETSE

A computational method for end-to-end intelligent prediction from raw protein sequence information to effector protein recognition.一种从原始蛋白质序列信息到效应蛋白识别的端到端智能预测的计算方法。


## Overview
MoCETSE enables end-to-end prediction of effector proteins from raw protein sequences through the following key steps:MoCETSE 通过以下关键步骤实现了从原始蛋白质序列到效应蛋白的端到端预测：
- Converts raw amino acid sequences into feature vector representations using the pre-trained protein language model ESM-1b.
- Generates more expressive sequence representations via a target preprocessing network based on hybrid convolutional experts.
- Introduces relative positional encoding in the Transformer layer to explicitly model relative distances between residues, achieving high-precision prediction of secreted proteins.


## Environment Requirements
Install the required dependencies before using MoCETSE:
```
python==3.9.7
torch==1.10.2   火炬= = 1.10.2
biopython==1.79
einops==0.4.1
fair-esm>=0.4.0
tqdm==4.64.0
numpy==1.21.2
scikit-learn==0.23.2   scikit-learn 版本 0.23.2
matplotlib==3.5.1
seaborn==0.11.0
tensorboardX==2.0
umap-learn==0.5.3   4.5.3 umap -learn = = 0
warmup-scheduler==0.3.2
```


## Model Training   ##模型培训
Run the following command to train the model (5-fold cross-validation):运行以下命令来训练模型（5 折交叉验证）：
```bash   ”“bash
for i in {0..4}   For I in {0..4}
do   做
python train.py  --model effectortransformer \
--data_dir data \   ——data_dir data
--lr 5e-5 \
--weight_decay 4e-5 \   --权重衰减 4e-5 \
--lr_scheduler cosine \   -- 学习率调度器余弦 \
--lr_decay_steps 30 \   -- 学习率衰减步数 30 \
--kfold 5 \   ——kfold 5 \
--fold_num $i \   ——fold_num $i \
--log_dir model   ——log_dir模型
done   完成
```

**Parameters explanation**:**参数说明**：
- `--model`: Model type (default: `effectortransformer`)- `--model`：模型类型（默认值：`effectortransformer`）
- `--data_dir`: Path to the training data directory`--data_dir`：训练数据目录的路径
- `--lr`: Learning rate (default: `5e-5`)- `--lr`：学习率（默认值：`5e-5`）
- `--weight_decay`: Weight decay (default: `4e-5`)`--weight_decay`：权重衰减（默认值：`4e-5`）
- `--lr_scheduler`: Learning rate scheduler (default: `cosine`)`--lr_scheduler`：学习率调度器（默认值：`余弦`）
- `--lr_decay_steps`: Learning rate decay steps (default: `30`)`--lr_decay_steps`：学习率衰减步数（默认值：`30`）
- `--kfold`: Number of cross-validation folds (default: `5`)`--kfold`：交叉验证的折数（默认值：`5`）
- `--fold_num`: Current fold index (in loop: `0..4`)
- `--log_dir`: Directory to save training logs and checkpoints


## Prediction
Use the trained model for effector protein prediction with:
```bash
python predict.py --fasta_path exmples/Test.fasta \
               --model_location checkpoint.pt \
               --secretion I II III IV VI \
               --out_dir results
```

**Parameters explanation**:
- `--fasta_path`: Path to the input FASTA file (e.g., `exmples/Test.fasta`)
- `--model_location`: Path to the trained model checkpoint (e.g., `checkpoint.pt`)
- `--secretion`: Types of secretion systems to predict (e.g., `I II III IV VI`)
- `--out_dir`: Directory to save prediction results


## Model Weight
The pre-trained MoCETSE model weight can be downloaded from:  
[https://drive.google.com/file/d/1J-E4FZmf-meSNSsjZ-96EVYmZwVICAYb/view?usp=sharing](https://drive.google.com/file/d/1J-E4FZmf-meSNSsjZ-96EVYmZwVICAYb/view?usp=sharing)
