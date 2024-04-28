# SeCor: Aligning Semantic and Collaborative Representations by Large Language Models for Next-Point-of-Interest Recommendations

## Overview

This work is currently under review, and more complete dataset will be updated upon acceptance.

In this paper, we propose a next POI recommendation method that aligns Semantic representations and Collaborative representations (SeCor). Collaborative representations are inherent latent intention embeddings, therefore conveying special semantics. SeCor aligns collaborative representations with semantic representations, utilizing LoRA to enhance the alignment capability of LLMs, thereby achieving a superior integration. We conducted multiple comparative experiments and investigated the rationality of each component of SeCor.

## Env Setting

```
conda create -n [env name] pip

conda activate [env name]

pip install -r requirements.txt
```

## Dataset

We have updated processed data in the './data' folder and the source code to generate these data in './prepare' folder. You can use these code to transfer Gowalla or some other POI recommendation dataset. Due to the large file limit of GitHub, we cannot directly upload the train set of TKY and the Gowalla dataset. We will update all of them in other link for more convenience after acceptance.

## First Stage Tuning

```
cd ./pretrain_cf
python train_basemodel.py
```

Currently we have uploaded LightGCN as CF model in SCE. Please pay attention to our corresponding settings in utils.py. you can modify the corresponding parameters or change to a more advanced CF-based model.

## Second Stage Tuning

```
cd ../
bash ./shell/instruct.sh <CUDA_ID> <RANDOM_SEED> <DATASET>
```

Please change the particular settings in instruct.sh file, especially the directory.

## Evaluation

Inference stage generates result file according to the setting in 'evaluate.sh'.

```
bash ./shell/evaluate.sh <CUDA_ID> <OUTPUT_DIR> <DATASET>
```

The output_dir should be same as the setting in the 'instruct.sh'.
