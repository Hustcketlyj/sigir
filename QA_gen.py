
import os
for i in range(34):
    command='python Generate_QAs.py \
        --train_path ../data/low_resource_LR10_'+str(i)+'_QA.json \
        --dev_path ../data/fever_dev.processed.json \
        --data_split train \
        --entity_dict ../output/intermediate/entity_dict_train.json \
        --save_path ../output/intermediate/precompute_QAs_train_'+str(i)+'.json'
    os.system(command)

for i in range(34):
    command='python Claim_Generation.py \
    --split train \
    --train_path ../data/low_resource_LR10_'+str(i)+'_QA.json \
    --dev_path ../data/fever_train.processed.json \
    --entity_dict ../output/intermediate/entity_dict_train.json \
    --QA_path ../output/intermediate/precompute_QAs_train_'+str(i)+'.json \
    --QA2D_model_path ../dependencies/QA2D_model \
    --sense_to_vec_path ../dependencies/s2v_old \
    --save_path ../output/SUPPORTED_claims_'+str(i)+'.json \
    --claim_type SUPPORTED'
    os.system(command)

for i in range(34):
    command='python Claim_Generation.py \
    --split train \
    --train_path ../data/low_resource_LR10_'+str(i)+'_QA.json \
    --dev_path ../data/fever_train.processed.json \
    --entity_dict ../output/intermediate/entity_dict_train.json \
    --QA_path ../output/intermediate/precompute_QAs_train_'+str(i)+'.json \
    --QA2D_model_path ../dependencies/QA2D_model \
    --sense_to_vec_path ../dependencies/s2v_old \
    --save_path ../output/REFUTED_claims_'+str(i)+'.json \
    --claim_type REFUTED'
    os.system(command)