# Robust Dancer: Long-term 3D Dance Synthesis Using Unpaired Data



## Get started

#### dependencies required

- python >=3.7
- pytorch 
- smplx
- aistplusplus_api

#### Get the checkpoint

get trained checkpoint from: https://drive.google.com/drive/folders/1Z-kghUqJLEbjgzg2soHsmnWoM62r5EzL?usp=sharing 

#### Run the code

1. get the data from aistpp website: https://google.github.io/aistplusplus_dataset/

2. data preprocess

```
python data_preprocess/extract_aist_feature.py  # necessary for test
python data_preprocess/motion_data_process.py  #necessary for train
```

3. run training

```
python train.py
```

4. run inference and test

```
python inference.py
python test_scripts/calculate_beat_scores.py
python test_scripts/calculate_fid_scores.py
python test_scripts/train_style_classi.py
python test_scripts/style_classi_new.py
```

## Citation