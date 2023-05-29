import jsonlines
import json


for i in range(34):
  f=jsonlines.open('low_resource_LR10_'+str(i)+'_DA_allen.jsonl')
  data=[line for line in f.iter()]
  result=[]
  for item in data:
    if item['label'] in [10,11,12]:
      item['context'] = item.pop('LM')
      result.append(item)
  print(len(result))
  json_object = json.dumps(result)
  new_filename='../Zero-shot-Fact-Verification/data/low_resource_LR10_'+str(i)+'_QA.json'
  with open(new_filename, "w") as outfile:
    outfile.write(json_object)
  outfile.close()