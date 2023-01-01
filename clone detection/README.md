# Clone Detection
## Data Preprocess
We follow the setting of **[Clone Detection](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Clone-detection-POJ-104)** from CodeXGLUE
to test the performance of ContraBERT_C and ContraBERT_G for fair comparison.
Our pre-trained models ContraBERT_C and
ContraBERT_G are available at **[ContraBERT_C](https://drive.google.com/drive/u/1/folders/1F-yIS-f84uJhOCzvGWdMaOeRdLsVWoxN)** and **[ContraBERT_G](https://drive.google.com/drive/u/1/folders/1t8VX6aYchpJolbH4mkhK3IQGzyHrDD3C)**.
Once the pre-trained models are download, it is suggested to put them to the folder "./saved_models/pretrain_models/".
## Fine-Tune
We fine-tuned the model on 2*V100-32G GPUs.
```shell   
Pretrain_dir=./saved_models/pretrain_models/
Model_type=ContraBERT_C   # ContraBERT_G
OUT_DIR=./saved_models/finetune_models/clone-detection/${Model_type}
mkdir -p $OUT_DIR
python clone_detection.py \
      --output_dir=${OUT_DIR} \
      --model_type=roberta \
      --tokenizer_name=microsoft/codebert-base \
      --model_name_or_path=${Pretrain_dir}/$Model_type \
      --do_train \
      --train_data_file=./data/finetune_data/clone_detection/train.jsonl \
      --eval_data_file=./data/finetune_data/clone_detection/valid.jsonl \
      --test_data_file=./data/finetune_data/clone_detection/test.jsonl \
      --epoch 2 \
      --block_size 400 \
      --train_batch_size 8 \
      --eval_batch_size 16 \
      --learning_rate 2e-5 \
      --max_grad_norm 1.0 \
      --evaluate_during_training \
      --seed 123456 2>&1| tee ${OUT_DIR}/train.log
```
## Inference and Evaluation
```shell  
  python clone_detection.py \
      --output_dir=${OUT_DIR} \
      --model_type=roberta \
      --tokenizer_name=microsoft/codebert-base \
      --model_name_or_path=${Pretrain_dir}/$Model_type \
      --do_eval \
      --do_test \
      --train_data_file=./data/finetune_data/clone_detection/train.jsonl \
      --eval_data_file=./data/finetune_data/clone_detection/valid.jsonl \
      --test_data_file=./data/finetune_data/clone_detection/test.jsonl \
      --epoch 2 \
      --block_size 400 \
      --train_batch_size 8 \
      --eval_batch_size 16 \
      --learning_rate 2e-5 \
      --max_grad_norm 1.0 \
      --evaluate_during_training \
      --seed 123456 2>&1| tee ${OUT_DIR}/test.log
```

## Result

| Model         | Accuracy | 
|---------------|--------|
| code2vec      | 1.98   | 
| NCC           | 54.19  |
| Aroma         | 55.12  | 	
| MISIM-GNN     | 82.45	 | 
| RoBERTa       | 79.96  |
| CodeBERT      | 84.29	 |
| GraphCodeBERT | 85.16	 |
| ContraBERT_C  | <strong>90.46</strong>	 |
| ContraBERT_G  | 90.06	 |
