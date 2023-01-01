# Code Search
## Data Preprocess
We follow the setting of **[GraphCodeBERT](https://github.com/microsoft/CodeBERT/tree/master/GraphCodeBERT/codesearch)**
to test the performance of ContraBERT_C and ContraBERT_G for fair comparison.
Our pre-trained models ContraBERT_C and
ContraBERT_G are available at **[ContraBERT_C](https://drive.google.com/drive/u/1/folders/1F-yIS-f84uJhOCzvGWdMaOeRdLsVWoxN)** and **[ContraBERT_G](https://drive.google.com/drive/u/1/folders/1t8VX6aYchpJolbH4mkhK3IQGzyHrDD3C)**.
Once the pre-trained models are download, it is suggested to put them to the folder "./saved_models/pretrain_models/".
## Fine-Tune
We fine-tuned the model on 2*V100-32G GPUs.
```shell   
lang=java # ruby, javascript, go, python, java, php
Pretrain_dir=./saved_models/pretrain_models/
Model_type=ContraBERT_C   # ContraBERT_G
OUT_DIR=./saved_models/finetune_models/code-search/$Model_type/$lang
mkdir -p $OUT_DIR
python code_search.py \
    --output_dir=${OUT_DIR} \
    --config_name=microsoft/graphcodebert-base \
    --model_name_or_path=${Pretrain_dir}/$Model_type \
    --tokenizer_name=microsoft/graphcodebert-base \
    --lang=$lang \
    --do_train \
    --train_data_file=./data/finetune_data/code_search/$lang/train.jsonl \
    --eval_data_file=./data/finetune_data/code_search/$lang/valid.jsonl \
    --test_data_file=./data/finetune_data/code_search/$lang/test.jsonl \
    --codebase_file=./data/finetune_data/code_search/$lang/codebase.jsonl \
    --num_train_epochs 10 \
    --code_length 256 \
    --data_flow_length 0 \
    --nl_length 128 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --seed 123456 2>&1| tee ${OUT_DIR}/train.log
```
## Inference and Evaluation
```shell  
python code_search.py \
    --output_dir=${OUT_DIR} \
    --config_name=microsoft/graphcodebert-base \
    --model_name_or_path=${Pretrain_dir}/$Model_type \
    --tokenizer_name=microsoft/graphcodebert-base \
    --lang=$lang \
    --do_eval \
    --do_test \
    --train_data_file=./data/finetune_data/code_search/$lang/train.jsonl \
    --eval_data_file=./data/finetune_data/code_search/$lang/valid.jsonl \
    --test_data_file=./data/finetune_data/code_search/$lang/test.jsonl \
    --codebase_file=./data/finetune_data/code_search/$lang/codebase.jsonl \
    --num_train_epochs 10 \
    --code_length 256 \
    --data_flow_length 0 \
    --nl_length 128 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --seed 123456 2>&1| tee ${OUT_DIR}/test.log
```

## Result
| Model          | Ruby           | Javascript     | Go             | Python         | Java           | PHP                      | Overall        |
|----------------|----------------|----------------|----------------|----------------|----------------|--------------------------|----------------|
| NBow           | 0.162          | 0.157          | 0.330          | 0.161          | 0.171          | 0.152                    | 0.189          |
| CNN            | 0.276          | 0.224          | 0.680          | 0.242          | 0.263          | 0.260                    | 0.324          |
| BiRNN          | 0.213          | 0.193          | 0.688          | 0.290          | 0.304          | 0.338                    | 0.338          | 	
| selfAtt        | 0.275          | 0.287          | 0.723          | 0.398          | 0.404          | 0.426                    | 0.419          |                     
| RoBERTa        | 0.587          | 0.517          | 0.850          | 0.587          | 0.599          | 0.560                    | 0.617          |
| RoBERTa (code) | 0.628          | 0.562          | 0.859          | 0.610          | 0.620          | 0.579                    | 0.643	         | 
| CodeBERT       | 0.679          | 0.620          | 0.882          | 0.672          | 0.676          | 0.628                    | 0.693          |
| GraphCodeBERT  | 0.703          | 0.644          | 0.897          | 0.692          | 0.691          | <strong> 0.649 </strong> | 0.713          |
| ContraBERT_C   | 0.688          | 0.626          | 0.892          | 0.678          | 0.685          | 0.634                    | 0.701          |
| ContraBERT_G   | <strong> 0.723</strong> | <strong> 0.656 </strong>| <strong> 0.899</strong> | <strong> 0.695 </strong>| <strong> 0.695 </strong>| <strong> 0.648  </strong>         | <strong> 0.719 </strong>|
