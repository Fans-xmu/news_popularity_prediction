# news_popularity_prediction
For news popularity prediction

## Dependency:
- python>=3.6
- torch>=1.6.0
- dgl>=0.6.1
- jieba>=0.42.1

## Data
The data is available at https://pan.baidu.com/s/13zmxj7o-oRRQXFcVOOiNYw, The extraction code is haju.

The code of F4 model we proposed consists of two parts. data process and model train.

## data process
we use the code in data_process folder
run 1_disa_csvreader.py, 2_vocab_news_creat.py, 3_entitiy_named.py, 4_creat_dataset.py, 5_calw2sTFIDF.py in Chinese datasets.
run 1_mind_csvreader.py, 2_vocab_news_creat.py, 3_entitiy_named.py, 4_creat_dataset.py, 5_calw2sTFIDF.py in MIND datasets.

## train
`python fhgnntrain.py`
## test
`testfhgnn.py`
