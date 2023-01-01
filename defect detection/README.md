# Defect Detection
## Data Preprocess
We follow the setting of **[Defect Detection](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Defect-detection)** from CodeXGLUE
to test the performance of ContraBERT_C and ContraBERT_G for fair comparison.
Our pre-trained models ContraBERT_C and
ContraBERT_G are available at **[ContraBERT_C](https://drive.google.com/drive/u/1/folders/1F-yIS-f84uJhOCzvGWdMaOeRdLsVWoxN)** and **[ContraBERT_G](https://drive.google.com/drive/u/1/folders/1t8VX6aYchpJolbH4mkhK3IQGzyHrDD3C)**.
Once the pre-trained models are download, it is suggested to put them to the folder "./saved_models/pretrain_models/".
## Fine-Tune
We fine-tuned the model on 2*V100-32G GPUs. 
```shell   
Pretrain_dir=./saved_models/pretrain_models/
Model_type=ContraBERT_C   # ContraBERT_G
OUT_DIR=./saved_models/finetune_models/vulnerability_detection/${Model_type}
mkdir -p ${OUT_DIR}
python vulnerability_detection.py \
      --output_dir=${OUT_DIR} \
      --model_type=roberta \
      --tokenizer_name=microsoft/codebert-base \
      --model_name_or_path=${Pretrain_dir}/$Model_type \
      --do_train \
      --train_data_file=./data/finetune_data/c_vulnerability/devign/train.jsonl \
      --eval_data_file=./data/finetune_data/c_vulnerability/devign/valid.jsonl \
      --test_data_file=./data/finetune_data/c_vulnerability/devign/test.jsonl  \
      --epoch 5 \
      --block_size 400 \
      --train_batch_size 32 \
      --eval_batch_size 64 \
      --learning_rate 2e-5 \
      --max_grad_norm 1.0 \
      --evaluate_during_training \
      --seed 123456 2>&1| tee ${OUT_DIR}/train.log
```
## Inference and Evaluation
```shell  
  python vulnerability_detection.py \
      --output_dir=${OUT_DIR} \
      --model_type=roberta \
      --tokenizer_name=microsoft/codebert-base \
      --model_name_or_path=${Pretrain_dir}/$Model_type \
      --do_eval \
      --do_test \
      --train_data_file=./data/finetune_data/c_vulnerability/devign/train.jsonl \
      --eval_data_file=./data/finetune_data/c_vulnerability/devign/valid.jsonl \
      --test_data_file=./data/finetune_data/c_vulnerability/devign/test.jsonl   \
      --epoch 5 \
      --block_size 400 \
      --train_batch_size 32 \
      --eval_batch_size 64 \
      --learning_rate 2e-5 \
      --max_grad_norm 1.0 \
      --evaluate_during_training \
      --seed 123456 2>&1| tee ${OUT_DIR}/test.log
```

## Result


| Model         | Acc    | 
|---------------|--------|
| BiLSTM         | 59.37  |
| TextCNN          | 60.69  | 
| RoBERTa   | 61.05  | 
| CodeBERT     | 62.08	 | 
| GraphCodeBERT | 62.85  | 
| ContraBERT_C  |  <strong>64.17</strong>	 | 
| ContraBERT_G  | 63.32	 | 
