# kG-CMI

This is the official implementation of "Knowledge graph enhanced cross-Mamba
interaction for medical visual question answering".

## Table of Contents
- [Requirements](#requirements)
- [Download](#download-m3ae)
- [Downstream Evaluation](#downstream-evaluation)

## Requirements
Run the following command to install the required packages:
```bash
pip install -r requirements.txt
```

## Download
You can download the pre-trained models from [here](https://drive.google.com/drive/folders/1lS2fm8RqFxoKQ8cgYD9hMFoOuHuKZNoY?usp=sharing).

## Downstream Evaluation
### 1. Dataset Preparation
Please organize the fine-tuning datasets as the following structure:
```angular2
root:[data]
+--finetune_data
| +--slack
| | +--train.json
| | +--validate.json
| | +--test.json
| | +--imgs
| +--vqa_rad
| | +--trainset.json
| | +--valset.json
| | +--testset.json
| | +--images
| +--ovqa
| | +--trainset.json
| | +--valset.json
| | +--testset.json
| | +--images
```

### 2. Pre-processing
Run the following command to pre-process the data:
```angular2
python prepro/prepro_finetuning_data.py
```
to get the following arrow files:
```angular2
root:[data]
+--finetune_arrows
| +--vqa_vqa_rad_train.arrow
| +--vqa_vqa_rad_val.arrow
| +--vqa_vqa_rad_test.arrow
| +--vqa_slack_train.arrow
| +--vqa_slack_test.arrow
| +--vqa_slack_val.arrow
| +--vqa_ovqa_train.arrow
| +--vqa_ovqa_val.arrow
| +--vqa_ovqa_test.arrow
```

### 3. Fine-Tuning
Now you can start to fine-tune the KG-CMI model:
```angular2
bash run_scripts/finetune_kgcmi.sh
```

### 4. Test
You can also test our fine-tuned models directly:
```angular2
bash run_scripts/test_kgcmi.sh
```
NOTE: This is a good way to check whether your environment is set up in the same way as ours (if you can reproduce the same results).

## Acknowledgement
The code is based on [M3AE](https://github.com/zhjohnchan/M3AE).
We thank the authors for their open-sourced code and encourage users to cite their works when applicable.

## Citations

[//]: # (If M3AE is useful for your research, please consider citing:)

[//]: # (```angular2)

[//]: # (@inproceedings{chen2022m3ae,)

[//]: # (  title={Multi-Modal Masked Autoencoders for Medical Vision-and-Language Pre-Training},)

[//]: # (  author={Chen, Zhihong and Du, Yuhao and Hu, Jinpeng and Liu, Yang and Li, Guanbin and Wan, Xiang and Chang, Tsung-Hui},)

[//]: # (  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},)

[//]: # (  year={2022},)

[//]: # (  organization={Springer})

[//]: # (})

[//]: # (```)
