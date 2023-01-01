# Clone Translation
## Data Preprocess
We follow the setting of **[Code Translation](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/code-to-code-trans)** from CodeXGLUE
to test the performance of ContraBERT_C and ContraBERT_G for fair comparison.
Our pre-trained models ContraBERT_C and
ContraBERT_G are available at **[ContraBERT_C](https://drive.google.com/drive/u/1/folders/1F-yIS-f84uJhOCzvGWdMaOeRdLsVWoxN)** and **[ContraBERT_G](https://drive.google.com/drive/u/1/folders/1t8VX6aYchpJolbH4mkhK3IQGzyHrDD3C)**.
Once the pre-trained models are download, it is suggested to put them to the folder "./saved_models/pretrain_models/".
## Fine-Tune
We fine-tuned the model on 2*V100-32G GPUs. We only provide C# to Java translation script. Exchanging ".cs" file to ".java" file as well as "OUT_DIR" are for Java to C# translation.
```shell   
Pretrain_dir=./saved_models/pretrain_models/
Model_type=ContraBERT_C   # ContraBERT_G
OUT_DIR=./saved_models/finetune_models/code-to-code-trans/$Model_type/csharp2java
mkdir -p $OUT_DIR
lr=1e-4
batch_size=32
beam_size=10
source_length=320
target_length=256

python code_translation.py \
    --output_dir=${OUT_DIR} \
    --model_type=roberta \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=${Pretrain_dir}/$Model_type \
    --tokenizer_name=roberta-base \
    --do_train \
    --do_eval \
    --train_filename=./data/finetune_data/code_trans/train.java-cs.txt.cs,./data/finetune_data/code_trans/train.java-cs.txt.java \
    --dev_filename=./data/finetune_data/code_trans/valid.java-cs.txt.cs,./data/finetune_data/code_trans/valid.java-cs.txt.java \
    --max_source_length ${source_length} \
    --max_target_length ${target_length} \
    --beam_size ${beam_size} \
    --train_batch_size ${batch_size} \
    --eval_batch_size ${batch_size} \
    --learning_rate ${lr} \
    --train_steps 100000 \
    --eval_steps 5000 2>&1| tee ${OUT_DIR}/train.log
```
## Inference and Evaluation
```shell  
batch_size=64
python code_translation.py \
    --output_dir=${OUT_DIR} \
    --model_type=roberta \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=${Pretrain_dir}/$Model_type \
    --load_model_path=${OUT_DIR}/checkpoint-best-bleu/pytorch_model.bin \
    --tokenizer_name=roberta-base \
    --do_test \
    --dev_filename=./data/finetune_data/code_trans/valid.java-cs.txt.cs,./data/finetune_data/code_trans/valid.java-cs.txt.java \
    --test_filename=./data/finetune_data/code_trans/test.java-cs.txt.cs,./data/finetune_data/code_trans/test.java-cs.txt.java \
    --max_source_length ${source_length} \
    --max_target_length ${target_length} \
    --beam_size ${beam_size} \
    --eval_batch_size ${batch_size} \
    --learning_rate ${lr} 2>&1| tee ${OUT_DIR}/test.log
```


## Result
Java to C#:

| Model     | BLEU                    | Acc                   |
|-----------|-------------------------|-----------------------|
| Navie     | 18.54                   | 0.0                   |
| PBSMT     | 43.53                   | 12.5                  |
| Transformer | 55.84                   | 33.0                  | 	
| Roborta (code) | 77.46	                  | 56.1                  | 
| CodeBERT  | 79.92                   | 	59.0                 | 
| GraphCodeBERT | 80.58                   | 59.4                  |
| ContraBERT_C | 	 79.95                 | 59.0                  |
| ContraBERT_G | <strong>80.78</strong>	 | <strong>59.9</strong> |


C# to Java:


| Model          | BLEU                    | Acc                  |
|----------------|-------------------------|----------------------|
| Navie          | 18.69	                  | 0.0                  |
| PBSMT          | 40.06	                  | 16.1                 |
| Transformer    | 50.47	                  | 37.9                 | 	
| Roborta (code) | 71.99	                  | 57.9                 | 
| CodeBERT       | 72.14	                  | 58.0                 | 
| GraphCodeBERT  | 72.64                   | 58.8                 |
| ContraBERT_C   | 	75.92                  | 59.6                 |
| ContraBERT_G   | <strong>76.24</strong>	 | <strong>60.5</strong> |