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
from transformers import MarianMTModel, MarianTokenizer



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

def train(dataset,msg,i=None):
    print(msg)
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=3)
    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=eva_datasets,
    compute_metrics=compute_metrics,)
    trainer.train()
    return evaluateTrainer(trainer,eva_datasets,i)

def evaluateTrainer(trainer,evaData,i):
    predictions = trainer.predict(evaData)
    preds = np.argmax(predictions.predictions, axis=-1)
    if i:
        return [accuracy_score(predictions.label_ids,preds),f1_score(predictions.label_ids,preds, average='macro'),i]
    else:
        return [accuracy_score(predictions.label_ids,preds),f1_score(predictions.label_ids,preds, average='macro')]


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

def format_batch_texts(language_code, batch_texts):
    formated_bach = [">>{}<< {}".format(language_code, text) for text in batch_texts]
    return formated_bach

def perform_translation(batch_texts, model, tokenizer, language="de"):
    # Prepare the text data into appropriate format for the model
    formated_batch_texts = format_batch_texts(language, batch_texts)
    # Generate translation using model
    translated = model.generate(**tokenizer(formated_batch_texts, return_tensors="pt", padding=True))
    # Convert the generated tokens indices back into text
    translated_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    return translated_texts

def backtrans(data):
    first_model_name = 'Helsinki-NLP/opus-mt-en-de'
    first_model_tkn = MarianTokenizer.from_pretrained(first_model_name)
    first_model = MarianMTModel.from_pretrained(first_model_name)
    second_model_name = 'Helsinki-NLP/opus-mt-de-en'
    second_model_tkn = MarianTokenizer.from_pretrained(second_model_name)
    second_model = MarianMTModel.from_pretrained(second_model_name)
    total=[]
    for item in data:
        if item['label']>=10:
            total.append({'label':item['label']-10,'text':'[CLS] '+item['claim']+' [SEP] '+item['LM']+' [SEP]'})
            tmp=[item['claim']]
            translated_texts = perform_translation(tmp, first_model, first_model_tkn)
            back_translated_texts = perform_translation(translated_texts, second_model, second_model_tkn)
            if tmp[0].replace(' ','')!=back_translated_texts[0].replace(' ',''):
                total.append({'label':item['label']-10,'text':'[CLS] '+back_translated_texts[0]+' [SEP] '+item['LM']+' [SEP]'})
    return total



def sort(result,num=10):
    target=result['mvlm']
    tmp = sorted(target, key=lambda d: d[0])
    tmp=tmp[:num]
    mean_acc={'gold':0,'mvlm':0,'eda':0,'bt':0}
    mean_f1={'gold':0,'mvlm':0,'eda':0,'bt':0}
    for item in tmp:
        index=item[2]
        mean_acc['mvlm']+=item[0]
        mean_acc['gold']+=result['gold'][index][0]
        mean_acc['eda']+=result['eda'][index][0]
        mean_acc['bt']+=result['bt'][index][0]
        mean_f1['mvlm']+=item[1]
        mean_f1['gold']+=result['gold'][index][1]
        mean_f1['eda']+=result['eda'][index][1]
        mean_f1['bt']+=result['bt'][index][1]
    print('**'*15)
    print('acc-gold:', mean_acc['gold']/num)
    print('acc-eda:', mean_acc['eda']/num)
    print('acc-mvlm:', mean_acc['mvlm']/num)
    print('acc-bt:', mean_acc['bt']/num)
    print('**'*5)
    print('f1-gold:', mean_f1['gold']/num)
    print('f1-eda:', mean_f1['eda']/num)
    print('f1-mvlm:', mean_f1['mvlm']/num)
    print('f1-bt:', mean_f1['bt']/num)


result={'gold':[],'mvlm':[],'eda':[],'backtrans':[]}

for i in range(20):
    data_DA='./low_resource_LR10_'+str(i)+'_DA_allen.jsonl'
    f=jsonlines.open(data_DA)
    data=[line for line in f.iter()]
    train_gold=[{'label':item['label']-10,'text':'[CLS] '+item['claim']+' [SEP] '+item['LM']+' [SEP]'} for item in data if item['label']>=10]
    train_mvlm=[{'label':item['label']%10,'text':'[CLS] '+item['claim']+' [SEP] '+item['LM']+' [SEP]'} for item in data]
    train_eda=EDA(data)
    train_backtrans=backtrans(data)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    train_gold=datasets.Dataset.from_pandas(pd.DataFrame(data=train_gold))
    train_gold_datasets = train_gold.map(tokenize_function, batched=True)
    train_eda=datasets.Dataset.from_pandas(pd.DataFrame(data=train_eda))
    train_eda_datasets = train_eda.map(tokenize_function, batched=True)
    train_mvlm=datasets.Dataset.from_pandas(pd.DataFrame(data=train_mvlm))
    train_mvlm_datasets = train_mvlm.map(tokenize_function, batched=True)
    train_backtrans=datasets.Dataset.from_pandas(pd.DataFrame(data=train_backtrans))
    train_backtrans_datasets = train_backtrans.map(tokenize_function, batched=True)
    result['gold'].append(train(train_gold_datasets,'gold only_'+str(i)))
    result['mvlm'].append(train(train_mvlm_datasets,'mvlm_'+str(i),i))
    result['eda'].append(train(train_eda_datasets,'eda_'+str(i)))
    result['backtrans'].append(train(train_backtrans_datasets,'backtrans_'+str(i)))
    print(result)

sort(result)

    
