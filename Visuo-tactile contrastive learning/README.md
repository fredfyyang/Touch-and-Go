# Visuo-tactile CMC
This repository contains the official PyTorch implementation of our application visuo-tactile CMC in the paper paper [Touch and Go: Learning from Human-Collected Vision and Touch](https://arxiv.org/pdf/2211.12498.pdf) .

## Environment
To setup the environment, please run

```bash
pip install -r requirements.txt
```

## Touch and Go Dataset
Data can be downloaded from [Touch and Go Dataset](https://drive.google.com/drive/folders/1NDasyshDCL9aaQzxjn_-Q5MBURRT360B).

### Preprocessing
- Convert Video into frames (within the dataset):
```bash
cd touch_and_go
python extract_frame.py
```

## Visuo-tactile CMC Training and Test
We provide training and evaluation scripts under `./scripts`, please check each bash file before running. Or you can run the code below:
- Pretraining
Example: To conduct visuo-tactile self-supervised learning for ResNet18, please run 
```bash
bash script/cmc_touch18.sh  
```
The checkpoints will be stored at `./ckpt/cmc`

- Downstream task
Example: Material Classification for ResNet18 linear probing pretrained from visuo-tactile cmc
```bash
bash material_cmc_touch18.sh
```
The checkpoints will be stored at `./ckpt/cls/`

Example: Hard/soft classification for ResNet18 linear probing pretrained from visuo-tactile cmc
```bash
bash hard_cmc_touch18.sh
```
The checkpoints will be stored at `./ckpt/hard/`

Example: Rough/smooth classification for ResNet18 linear probing pretrained from visuo-tactile cmc
```bash
bash rough_cmc_touch18.sh
```
The checkpoints will be stored at `./ckpt/rough/`

## Todo
- [x] Code for visuo-tactile CMC pretraining
- [x] Code for downstream tasks on Touch and Go dataset (i.e. material classification, hard/soft and rough/smooth classification)
- [ ] Pretrained model via our pretraining

### Citation
If you use this code for your research, please cite our [paper](hhttps://arxiv.org/pdf/2211.12498.pdf).
```
@inproceedings{
yang2022touch,
  title={Touch and Go: Learning from Human-Collected Vision and Touch},
  author={Fengyu Yang and Chenyang Ma and Jiacheng Zhang and Jing Zhu and Wenzhen Yuan and Andrew Owens},
  booktitle={Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2022}
}
```

### Acknowledgments
We thank Xiaofeng Guo and Yufan Zhang for the extensive help with the GelSight sensor, and thank Daniel Geng, Yuexi Du and Zhaoying Pan for the helpful discussions. This work was supported in part by Cisco Systems and Wang Chu Chien-Wen Research Scholarship. This application is based on the code of [Contrastive Multiview Coding](https://github.com/HobbitLong/CMC/tree/master)(ECCV 2020). 
