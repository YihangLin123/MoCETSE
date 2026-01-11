# MoCETSE

We have developed a new computational method called MoCETSE, which enables end-toend intelligent prediction from raw protein sequence information to effector protein recognition. Specifically, MoCETSE first converts raw amino acid sequences into feature vector representations through the pre-trained protein language model ESM-1b. Next, it uses a target preprocessing network based on hybrid convolutional experts to generate more expressive sequence representations. In the Transformer layer, MoCETSE introduces relative positional encoding, explicitly models the relative distances between residues, and achieves high-precision prediction of secreted proteins.

![MoCETSE 模型架构](model%20framework.png)

## Environment Requirements
Install the required dependencies before using MoCETSE:
```
python==3.9.7
torch==1.10.2
biopython==1.79
einops==0.4.1
fair-esm>=0.4.0
tqdm==4.64.0
numpy==1.21.2
scikit-learn==0.23.2
matplotlib==3.5.1
seaborn==0.11.0
tensorboardX==2.0
umap-learn==0.5.3
warmup-scheduler==0.3.2
```

## Model Weight
### Download Link
The pre-trained MoCETSE model weight file is available for download via Google Drive:  
[Model](https://drive.google.com/file/d/17s9Ybz8bqCbaqsB-yOAn68v6kxrpWvh_/view?usp=sharing)

## Model Training
Run the following command to train the model (5-fold cross-validation):
```bash
for i in {0..4}
do
   python train.py  --model effectortransformer \
   --data_dir data \
   --lr 5e-5 \
   --weight_decay 4e-5 \
   --lr_scheduler cosine \
   --lr_decay_steps 30 \
   --kfold 5 \
   --fold_num $i \
   --log_dir model
done
```

## Prediction
Use the trained model for effector protein prediction with:
```bash
python predict.py --fasta_path exmples/Test.fasta \
                  --model_location checkpoint.pt \
                  --secretion I II III IV VI \
                  --out_dir results
```

## Genome‑wide prediction of secreted proteins
```bash
python predict_genome.py --fasta_path genome_data/NC_002942.5_protein.fasta \
			--model_location checkpoint.pt \
			--data_dir data \
			--out_dir results/NC_002942.5
```

### Install the logomaker Package
Before plotting sequence attention, you need to install the `logomaker` package first. Run the following command in your terminal:
```bash
pip install logomaker
```

### Install Required Packages
To predict secretion systems and substrate proteins, you need to install `macsyfinder` and `hmmer` first. Execute the following commands according to the package type:
#### Install macsyfinder (PyPI)
```bash
pip install macsyfinder
conda install -c bioconda hmmer
cd data
unzip TXSS.zip
```
