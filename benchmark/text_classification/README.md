# ALAM text classification
## Requirements
We conducted experiments in the torch 1.10.1 environment. This environment with requirements is available at: 
```bash
conda env create -f alam_glue.yaml
```
And then, install ALAM at [alam](https://github.com/KH9NHAKRFF/ALAM).

## Finetune models 
### Benchmark accuracy
```bash
python run_glue.py --model_name_or_path ARCH
--task_name TASK --max_length 128 --per_device_train_batch_size 16 --per_device_eval_batch_size 128 --learning_rate 5e-5 --num_train_epochs 3 --pad_to_max_length  --output_dir log/TASK/ --alam --bit BIT --save_bit_dir log/TASK/ --load_bit_dir log/TASK/
```
The choices for TASK are {cola, mnli, mrpc, qnli, qqp, rte, sst2, stsb, wnli}. 

The choices for ARCH are defined in huggingface.co/models. In the paper, we experiment with ```distilbert-base-uncased```, ```bert-base-cased```, and ```bert-large-cased```.

To apply alam, add ```--alam``` and ```--bit BIT```, choosing a average bit of activations. In the paper, we experimented with (2, 1.5, 1). 

Note that you can save the **alocated bit tensor** for activations by ```--save_bit_dir DIR```. Once you obtain the bit tensor for a certain model, you can freely reuse it in other scenarios (i.e. another learning rate or task) where you are fine-tuning the same model by ```--load_bit_dir DIR```, eliminating the need to recalculate sensitivity and assign bits once again. 

### Benchmark memory
Add `--get-mem` to the end of the command. For example, to get the training memory of auto precision 4 bit on bert-large-cased sst2 dataset, run the following command:
```bash
python run_glue.py --model_name_or_path ARCH
--task_name TASK --max_length 128 --per_device_train_batch_size 16 --per_device_eval_batch_size 128 --learning_rate 5e-5 --num_train_epochs 3 --pad_to_max_length  --output_dir log/TASK/LEVEL/ --alam --bit BIT --get-mem
```







