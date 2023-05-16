from huggingface_hub import login
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer
login('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')
from datasets import load_dataset
import json
import secrets
import spacy
nlp=spacy.load('en_core_web_sm')
import spacy
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import jsonlines
from transformers import DataCollatorForLanguageModeling
from transformers.data.data_collator import tf_default_data_collator
import collections
import numpy as np
from transformers import TrainingArguments
from transformers import Trainer
from transformers import default_data_collator
import math
import spacy
from transformers import pipeline
nlp=spacy.load('en_core_web_sm')

def has_noun(sent):
    doc = nlp(sent)
    word=[]
    doc=[i for i in doc]
    for token in doc:
        if 'NN' in token.tag_:
          return True
    return False
            
def select_sample(item):
  doc=nlp(item['claim'])
  ents=[i for i in doc.ents if i.label_!='PERSON']
  if len(ents)!=0:
    return True
  return False
#all=select_sample(all)

def subsample(filename='train_selected_person_nn.json',n=20):
  print('Sub sampling ... ')
  f = open('fever_train_processed.json')
  evidence = json.load(f)
  evidence=json2dict(evidence)
  f = open(filename)
  all = json.load(f)
  data_0=[['SUPPORTED '+line['claim']+' SUPPORTED'+ ' [SEP] '+evidence[line['id']],line['id']] for line in all if line['label'].upper()=='SUPPORTS']
  data_1=[['REFUTED '+line['claim']+' REFUTED'+ ' [SEP] '+evidence[line['id']],line['id']] for line in all if line['label'].upper()=='REFUTES']
  data_2=[['NOT ENOUGH INFORMATION '+line['claim']+' NOT ENOUGH INFORMATION',line['id']] for line in all if line['label'].upper()=='NOT ENOUGH INFO']
  print(len(data_0),len(data_1),len(data_2))
  for i in range(n):
    secure_random = secrets.SystemRandom()      # creates a secure random object.
    num_to_select = 10               # set the number to select here.
    random_0 = secure_random.sample(data_0, num_to_select)
    random_1 = secure_random.sample(data_1, num_to_select)
    random_2 = secure_random.sample(data_2, num_to_select)
    data = [{'id':i[1],'text':i[0]} for i in random_0+random_1+random_2]
    json_object = json.dumps(data)
    new_filename='low_resource_LR10_'+str(i)+'_pretuned.json'
    with open(new_filename, "w") as outfile:
      outfile.write(json_object)
    outfile.close()
  f.close()

def tokenize_function(examples):
    result = tokenizer(examples["text"],padding='max_length', max_length=128)
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result

def group_texts(examples):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result

def whole_word_masking_data_collator(features):
    for feature in features:
        #print('before',len(feature['input_ids']))
        #print(feature['input_ids'])
        word_ids = feature.pop("word_ids")
        tmp_id=feature["input_ids"]
        # Create a map between words and corresponding token indices
        anchor=None
        for idx, i_id in enumerate(tmp_id):
          if i_id==102:
            anchor=idx
            print('find pos:',anchor)
            break
        word_ids_remain=word_ids[anchor:]
        word_ids=word_ids[:anchor]
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        # Randomly mask words
        mask = np.random.binomial(1, wwm_probability, (len(mapping),))
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        input_ids_remain=input_ids[anchor:]
        input_ids=input_ids[:anchor]
        new_labels = [-100] * len(labels)
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = tokenizer.mask_token_id
        input_ids+=input_ids_remain
        feature["input_ids"]=input_ids
        feature["labels"] = new_labels
        word_ids+=word_ids_remain
        #print('after',len(feature['input_ids']))
    return default_data_collator(features)

def NerMask(sentence,unmasker,evidence):#label is supported
  text=sentence.replace('SUPPORTED ','')
  text=text.replace(' SUPPORTED','')
  text=text.replace('REFUTED ','')
  text=text.replace(' REFUTED','')
  text=text.replace('NOT ENOUGH INFORMATION ','')
  text=text.replace('NOT ENOUGH INFORMATION ','')
  doc=nlp(text)
  ents=[i for i in doc.ents if i.label_!='PERSON']
  aug=[]
  if len(ents)==0:
    return []
  for tmp in ents:
    start_pos=tmp.start_char
    end_pos=tmp.end_char
    masked_sent=text[:start_pos]+'[MASK]'+text[end_pos:]
    masked_sent_0='SUPPORTED '+masked_sent+' SUPPORTED'+' [SEP] '+evidence
    DA_0=unmasker(masked_sent_0)[0]['sequence']
    entity_0=nlp(DA_0)
    entity_0=[i for i in entity_0.ents if i.label_!='PERSON']
    masked_sent_1='REFUTED '+masked_sent+' REFUTED'+' [SEP] '+evidence
    DA_1=unmasker(masked_sent_1)[0]['sequence']
    entity_1=nlp(DA_1)
    entity_1=[i for i in entity_1.ents if i.label_!='PERSON']
    masked_sent_2='NOT ENOUGH INFORMATION '+masked_sent+' NOT ENOUGH INFORMATION'+' [SEP] '+evidence
    DA_2=unmasker(masked_sent_2)[0]['sequence']
    entity_2=nlp(DA_2)
    entity_2=[i for i in entity_2.ents if i.label_!='PERSON']
    DA_0=DA_0.split(' [SEP] ')[0]
    DA_1=DA_1.split(' [SEP] ')[0]
    DA_2=DA_2.split(' [SEP] ')[0]
    DA_0=DA_0.replace('SUPPORTED ','')
    DA_1=DA_1.replace('REFUTED ','')
    DA_0=DA_0.replace(' SUPPORTED','')
    DA_1=DA_1.replace(' REFUTED','')
    DA_2=DA_2.replace('NOT ENOUGH INFORMATION ','')
    DA_2=DA_2.replace(' NOT ENOUGH INFORMATION','')
    if DA_1.replace(' ','')!=text.replace(' ','') and len(entity_1)>=len(ents):
      aug.append('F:'+DA_1)
    if DA_0.replace(' ','')!=text.replace(' ','') and DA_0.replace(' ','')!=DA_1.replace(' ','') and len(entity_0)>=len(ents):
      aug.append('T:'+DA_0)
    #if DA_2.replace(' ','')!=text.replace(' ','') and DA_2.replace(' ','')!=DA_1.replace(' ','') and DA_2.replace(' ','')!=DA_0.replace(' ',''):
    #  aug.append('N:'+DA_2)
  return aug

def PosMask(sentence,unmasker,evidence):
  text=sentence.replace('SUPPORTED ','')
  text=text.replace(' SUPPORTED','')
  text=text.replace('REFUTED ','')
  text=text.replace(' REFUTED','')
  text=text.replace('NOT ENOUGH INFORMATION ','')
  text=text.replace(' NOT ENOUGH INFORMATION','')
  doc = nlp(text)
  word=[]
  doc=[i for i in doc]
  doc_onehot=['NN' in token.tag_ and 'NN' not in token.tag_ for token in doc]
  aug=[]
  for i in range(len(doc)):
      if doc_onehot[i] and len(doc)!=0:
        masked_sent_0='SUPPORTED '+' '.join([token.text for token in doc][:i])+' [MASK] '+' '.join([token.text for token in doc][i+1:])+' SUPPORTED'+' [SEP] '+evidence
        DA_0=unmasker(masked_sent_0)[0]['sequence']
        masked_sent_1='REFUTED '+' '.join([token.text for token in doc][:i])+' [MASK] '+' '.join([token.text for token in doc][i+1:])+' REFUTED'+' [SEP] '+evidence
        DA_1=unmasker(masked_sent_1)[0]['sequence']
        DA_0=DA_0.split(' [SEP] ')[0]
        DA_1=DA_1.split(' [SEP] ')[0]
        DA_0=DA_0.replace('SUPPORTED ','')
        DA_1=DA_1.replace('REFUTED ','')
        DA_0=DA_0.replace(' SUPPORTED','')
        DA_1=DA_1.replace(' REFUTED','')
        if DA_1.replace(' ','')!=text.replace(' ',''):
          aug.append('F:'+DA_1)
        if DA_0.replace(' ','')!=text.replace(' ','') and DA_0.replace(' ','')!=DA_1.replace(' ',''):
          aug.append('T:'+DA_0)
  
  return aug

def da(filename,i):
  f = open(filename)
  original_data = json.load(f)
  unmasker=pipeline(
    "fill-mask", model="jojoUla/bert-large-cased-sigir-support-refute-no-label-40-2nd-test-LR10-8-fast-"+str(i)
  )
  with open('./low_resource_LR10_'+str(i)+'_DA.jsonl','w') as outfile:
    for item in original_data:
      text=item['text']
      evidence=text.split(' [SEP] ')[1]
      text=text.split(' [SEP] ')[0]
      id=item['id']
      if 'SUPPORTED' in text:
        label=0
      elif 'REFUTED' in text:
        label=1
      else:
        label=2
      text=text.replace('SUPPORTED ','')
      text=text.replace(' SUPPORTED','')
      text=text.replace('REFUTED ','')
      text=text.replace(' REFUTED','')
      text=text.replace('NOT ENOUGH INFORMATION ','')
      text=text.replace(' NOT ENOUGH INFORMATION','')
      tmp={'label':label+10,'claim':text,'id':item['id']}
      json.dump(tmp, outfile)
      outfile.write('\n')
      if label in [0,1,2]:
        DA_ner=NerMask(text,unmasker,evidence)
        DA_pos=PosMask(text,unmasker,evidence)
        for item in DA_ner+DA_pos:
          if 'F:' in item:
            tmp={'label':1,'claim':item.replace('F:',''),'id':id}
          elif 'T:' in item:
            tmp={'label':0,'claim':item.replace('T:',''),'id':id}
          else:
            tmp={'label':2,'claim':item.replace('N:',''),'id':id}
          json.dump(tmp, outfile)
          outfile.write('\n')

  f.close()
  outfile.close()

def json2dict(evidence):
   
   dic={}
   for item in evidence:
      dic[item['id']]=item['context']
   return dic

def allen(filename,i):
  #nlp=spacy.load('en_core_web_sm')
  f=jsonlines.open(filename)
  data=[line for line in f.iter()]
  f = open('fever_train_processed.json')
  evidence = json.load(f)
  evidence=json2dict(evidence)
  evidence_key=list(evidence.keys())
  evidence_values=list(evidence.values())
  with open('./low_resource_LR10_'+str(i)+'_DA_allen.jsonl','w') as outfile:
    for item in data:
      claim=item['claim']
      #doc=nlp(claim)
      #ents=[i for i in doc.ents if i.label_!='PERSON']
      #if len(ents)==0:
      #  premise='None'
      #else:
      #  start_pos=ents[-1].start_char
      #  end_pos=ents[-1].end_char
      #  masked_sent=claim[:start_pos]+'[MASK]'+claim[end_pos:]
      #  unmasked=unmasker0(masked_sent)[0]['sequence']
      if item['id'] in evidence_key:
         premise=evidence[item['id']]
      else:
         secure_random = secrets.SystemRandom()      # creates a secure random object.
         premise= secure_random.sample(evidence_values, 1)[0]      
      hypo=claim
      tmp={'label':item['label'],'claim':hypo,'LM':premise,'id':item['id']}
      json.dump(tmp, outfile)
      outfile.write('\n')
  outfile.close()


subsample()
#unmasker0=pipeline(
    #"fill-mask", model="jojoUla/bert-large-cased-sigir-support-refute-no-label-40"
#    "fill-mask", model="bert-large-cased"
##)
for i in range(20):
  model_checkpoint="jojoUla/bert-large-cased-sigir-support-refute-no-label-40"
  tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
  model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
  filename='low_resource_LR10_'+str(i)+'_pretuned.json'
  print('loading datasets...')
  imdb_dataset = load_dataset('json', data_files=filename)
  print('datasets loaded...')
  print('datasets tokenizing...')
  tokenized_datasets = imdb_dataset.map(
      tokenize_function, batched=True, remove_columns=["text",'id']
  )
  print('datasets tokenized...')
  chunk_size = 128
  lm_datasets = tokenized_datasets.map(group_texts, batched=True)
  print('datasets complete...')
  #lm_datasets
  data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
  wwm_probability = 0.15
  train_size = 28
  test_size = 2
  downsampled_dataset = lm_datasets["train"].train_test_split(
      train_size=train_size, test_size=test_size, seed=42
  )
  downsampled_dataset
  batch_size = 64
  logging_steps = (len(downsampled_dataset["train"]) // batch_size) +1
  model_name = model_checkpoint.split("/")[-1]
  print('loading training args...')
  training_args = TrainingArguments(
      output_dir=f"{model_name}-2nd-test-LR10-8-fast-"+str(i),
      overwrite_output_dir=True,
      evaluation_strategy="epoch",
      learning_rate=4e-5,
      num_train_epochs=8.0,
      weight_decay=0.01,
      per_device_train_batch_size=batch_size,
      per_device_eval_batch_size=batch_size,
      push_to_hub=True,
      logging_steps=logging_steps,
      remove_unused_columns=False,
      
  )
  print('loading trainer...')
  trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=downsampled_dataset["train"],
      eval_dataset=downsampled_dataset["test"],
      data_collator=whole_word_masking_data_collator,#data_collator,
      tokenizer=tokenizer,
  )
  print('original perplexity...')
  eval_results = trainer.evaluate()
  print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
  print('training...')
  trainer.train()
  eval_results = trainer.evaluate()
  print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
  trainer.push_to_hub()
  da(filename,i)
  allen('low_resource_LR10_'+str(i)+'_DA.jsonl',i)