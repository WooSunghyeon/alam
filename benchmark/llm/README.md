
# ALAM large lange models ️

In this repository, we applied our ALAM to [lit-llama](https://github.com/Lightning-AI/lit-llama) Before running our code, we recommend first executing this repository. Thanks the authors for open-source code.

## Requirements
We conducted experiments in the torch 2.0.1 environment. This environment with requirements is available at: 
```bash
conda env create -f alam_llm.yaml
```
And then, install ALAM at [alam](https://github.com/KH9NHAKRFF/ALAM).

## Prepared the pretrained model

Check the guideline [guide](howto/download_weights.md).

## Prepair data

Prepair [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) datasets by
```bash
   python scripts/prepare_alpaca.py
   ```


## Finetune the model by ALAM

Finetune the model by ALAM as below:

   ```bash
   python finetune/lora.py --pretrained_path PRETRAINED_PATH --data_dir DATA_DIR --tokenizer TOKENIZER_PATH --qlora False --do_alam True --bit 2 --save_bit_dir DIR --load_bit_dir DIR
   ```
   
In experiments, we apply parameter efficient fine-tuning (PEFT) such as [LoRA](https://arxiv.org/abs/2106.09685) and [QLoRA](https://arxiv.org/abs/2303.16199).

To apply qlora, use ```--qlora```. It automatically perform QLoRA with nf4 bit precision. 

To apply apla, add ```--do_alam``` and ```--bit BIT```, choosing a average bit of activations. In the paper, we experimented with (4, 3, 2, 1.5, 1).

Note that you can save the **alocated bit tensor** for activations by ```--save_bit_dir DIR```. Once you obtain the bit tensor for a certain model, you can freely reuse it in other scenarios (i.e. another learning rate or task) where you are fine-tuning the same model by ```--load_bit_dir DIR```, eliminating the need to recalculate sensitivity and assign bits once again. 
