# Tactile-driven image stylization (TDIS)

## Environment
To setup the environment, please simply run

```bash
conda env create -f environment.yml
conda activate TDIS
```

## Touch and Go Dataset
Data can be downloaded from [Touch and Go Dataset](https://drive.google.com/drive/folders/1NDasyshDCL9aaQzxjn_-Q5MBURRT360B).

### Preprocessing
- Convert Video into frames (within the dataset):
```bash
cd touch_and_go
python extract_frame.py
```

- Sample frames and train/test split(within the TDIS code):
```bash
cd TDIS
python datasets/touch_and_go/generate_train_test.py  
```
We have already provided train/test split of our implementation in the `./datasets/touch_and_go/` folder.

## TDIS Training and Test
We provide training and evaluation scripts under `./scripts`, please check each bash file before running. Or you can run the code below:
- Training
```bash
python train.py --dataroot path/to/the/dataset --name touch_and_go --dataset_mode touch_and_go --model TDIS 
```
The checkpoints will be stored at `./checkpoints/touch_and_go/`

- Test the model
```bash
python test.py --dataroot path/to/the/dataset --name touch_and_go --dataset_mode touch_and_go --model TDIS
```
The test results will be saved to a html file at `./results/touch_and_go/test_latest/index.html`

### Citation
If you use this code for your research, please cite our [paper](https://openreview.net/pdf?id=ZZ3FeSSPPblo).
```
@inproceedings{
yang2022touch,
  title={Touch and Go: Learning from Human-Collected Vision and Touch},
  author={Fengyu Yang and Chenyang Ma and Jiacheng Zhang and Jing Zhu and Wenzhen Yuan and Andrew Owens},
  booktitle={Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2022},
  url={https://openreview.net/forum?id=ZZ3FeSSPPblo}
}
```

### Acknowledgments
We thank Xiaofeng Guo and Yufan Zhang for the extensive help with the GelSight sensor, and thank Daniel Geng, Yuexi Du and Zhaoying Pan for the helpful discussions. This work was supported in part by Cisco Systems and Wang Chu Chien-Wen Research Scholarship.
