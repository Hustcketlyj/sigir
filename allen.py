#import nltk
#nltk.data.path.append('/research/d5/gds/yjlu22/cache/nltk_cache')
#from allennlp.predictors.predictor import Predictor
#import allennlp_models.tagging

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score
import jsonlines
import json
import pickle
import jsonlines
from datasets import load_dataset
import datasets
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import pandas as pd
from transformers import TrainingArguments
import numpy as np
import evaluate
import random
from transformers import TrainingArguments, Trainer



def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

data_DA='./fever_dev_processed_all.jsonl'
f=jsonlines.open(data_DA)
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
data=[line for line in f.iter()]
eva=[{'label':item['label'],'text':'[CLS] '+item['claim']+' [SEP] '+item['LM']+' [SEP]'} for item in data]
eva=datasets.Dataset.from_pandas(pd.DataFrame(data=eva))
eva_datasets = eva.map(tokenize_function, batched=True)
training_args = TrainingArguments(output_dir="test_trainer")
metric = evaluate.load("accuracy")
training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch",num_train_epochs=8.0)

def train(dataset,msg):
    print(msg)
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=3)
    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=eva_datasets,
    compute_metrics=compute_metrics,)
    trainer.train()

def EDA(data):
    total=[]
    for item in data:
        if item['label']>=10:
            total.append({'label':item['label']-10,'text':'[CLS] '+item['claim']+' [SEP] '+item['LM']+' [SEP]'})
            tmp=item['claim']
            tmp=tmp.split(' ')
            tmp.remove(random.choice(tmp))
            total.append({'label':item['label']-10,'text':'[CLS] '+' '.join(tmp)+' [SEP] '+item['LM']+' [SEP]'})
    return total

for i in range(20):
    data_DA='./low_resource_LR10_'+str(i)+'_DA_allen.jsonl'
    f=jsonlines.open(data_DA)
    data=[line for line in f.iter()]
    train_gold=[{'label':item['label']-10,'text':'[CLS] '+item['claim']+' [SEP] '+item['LM']+' [SEP]'} for item in data if item['label']>=10]
    train_mvlm=[{'label':item['label']%10,'text':'[CLS] '+item['claim']+' [SEP] '+item['LM']+' [SEP]'} for item in data]
    train_eda=EDA(data)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    train_gold=datasets.Dataset.from_pandas(pd.DataFrame(data=train_gold))
    train_gold_datasets = train_gold.map(tokenize_function, batched=True)
    train_eda=datasets.Dataset.from_pandas(pd.DataFrame(data=train_eda))
    train_eda_datasets = train_eda.map(tokenize_function, batched=True)
    train_mvlm=datasets.Dataset.from_pandas(pd.DataFrame(data=train_mvlm))
    train_mvlm_datasets = train_mvlm.map(tokenize_function, batched=True)
    train(train_gold_datasets,'gold only_'+str(i))
    train(train_mvlm_datasets,'mvlm_'+str(i))
    train(train_eda_datasets,'eda_'+str(i))