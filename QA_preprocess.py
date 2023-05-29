import jsonlines
import json


for i in range(34):
  f=jsonlines.open('low_resource_LR10_'+str(i)+'_DA_allen.jsonl')
  data=[line for line in f.iter()]
  for item in data:
    item['context'] = item.pop('LM')
  #print(data[:2])
  json_object = json.dumps(data)
  new_filename='../Zero-shot-Fact-Verification/data/low_resource_LR10_'+str(i)+'_QA.json'
  with open(new_filename, "w") as outfile:
    outfile.write(json_object)
  outfile.close()