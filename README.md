# Towards-Unbiased-Cross-Modal-Representation-Learning-for-Food-Image-to-Recipe-Retrieval
Implementation for our paper "Towards Unbiased Cross-Modal Representation Learning for Food Image-to-Recipe Retrieval"

[arXiv](https://arxiv.org/pdf/2511.15201)

## Data preparation

Here we use the recipe1M dataset for training and evaluation. You can download the data files [here](https://drive.google.com/drive/folders/1lHvTJMsPISkPEFlZIi8p0CcgKqggF5Fp?usp=drive_link). 

The data files are **train.pkl, val.pkl and test.pkl**. Each file is in dictionary format and contains the recipes, where recipe id is the key and recipe info (title, ingredients, instructions, dish image) are the value. 

**ing2label_topk500.pkl** contains the 500 most frequent ingredients in recipe1M, which is a dictionary with key as the original ingredient names and value as the corresponding ingredient label.

**id2labels_train.pkl, id2labels_val.pkl and id2labels_test.pkl ** contains the ingredient labels of each recipe for train, test and valiation datasets, which is a dictionary of recipe id as key and ingredient label list as the value.



## Training the baseline model HT

We use [HT](https://arxiv.org/pdf/2103.13061) as our baseline model. To train the model, using the following command:

```python
python train_retrieval.py --dist-url tcp://127.0.0.1:6001 --cls_weight 0.0 --retrieval_weight 1.0 --batch-size 64 --dataname recipe1M --backbone resnet50 --epochs 100 --img_size 224 --aug_type ret
```

You can find the training log in **"./logs/s1.out"**, and the checkpoint from [here](https://drive.google.com/drive/folders/1AvlBZ8ibncF1BBCmSXezKV73Cvczi_7m?usp=drive_link).

## Create the ingredient dictionary

Before training our prospoed debiasing method, the ingredient dictionary should be created first:

```python
python create_textual_dictionary.py --dist-url tcp://127.0.0.1:6002 --cls_weight 0.0 --retrieval_weight 1.0 --batch-size 64 --dataname recipe1M --backbone resnet50 --epochs 100 --img_size 224 --aug_type ret
```

You can also download the dictionry from [here](https://drive.google.com/drive/folders/1AvlBZ8ibncF1BBCmSXezKV73Cvczi_7m?usp=drive_link).

To evaluate the model, add ```--evaluate ``` to the above command, i.e.,

```python
python create_textual_dictionary.py --dist-url tcp://127.0.0.1:6002 --cls_weight 0.0 --retrieval_weight 1.0 --batch-size 64 --dataname recipe1M --backbone resnet50 --epochs 100 --img_size 224 --aug_type ret --evaluate
```



## Training the HT with Ingredient Debiasing

Run the following command to train the debiaisng model:

```python
python train_debiasing.py --dist-url tcp://127.0.0.1:6102 --cls_weight 0.001 --retrieval_weight 1.0 --batch-size 64 --dataname recipe1M --backbone resnet50 --epochs 100 --img_size 224 --aug_type ret
```

The training log is in **"./logs/s2_debiasing.out"**  and the checkpoint can be also downloaded from [here](https://drive.google.com/drive/folders/1PzLIcNR8hgcaY6vR83lF7F7r2UPIWC2-?usp=drive_link).

To evaluate the model, add ```--evaluate``` to the above command.



## Acknowledgement

Thanks the greate implementation of [query2labels](https://github.com/SlongLiu/query2labels) and [HT](https://github.com/amzn/image-to-recipe-transformers), where I inspired my implementation.
