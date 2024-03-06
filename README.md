# ALAM: Averaging for Low-Precision Activation for memory efficient training

This repository is the official implementation of https://openreview.net/forum?id=BG7H1XsMG0.

+ We propose Average Quantization, which generates high-quality sub-1b activations, enabling further compression without sacrificing accuracy. To the best of our knowledge, this is the first attempt to compress activations through simple averaging.
+ We propose GradNormVar, a lightweight sensitivity calculation algorithm that employs the variance of the L2 norm of parameter gradients, eliminating the need to retain all parameter gradients and substantially reducing memory usage.
+ The proposed ALAM framework is demonstrated to successfully compress activations in various transformer models, achieving a 12.5$\times$ compression rate in LLaMA2-7B.

## Abstract
 One of the key challenges in deep neural network training is the substantial amount of GPU memory required to store activations obtained in the forward pass. Various Activation-Compressed Training (ACT) schemes have been proposed to mitigate this issue; however, it is challenging to adopt those approaches in recent transformer-based large language models (LLMs), which experience significant performance drops when the activations are deeply compressed during training. In this paper, we introduce ALAM, a novel ACT framework that utilizes average quantization and a lightweight sensitivity calculation scheme, enabling large memory saving in LLMs while maintaining training performance. We first demonstrate that compressing activations into their group average values minimizes the gradient variance. Employing this property, we propose Average Quantization which provides high-quality deeply compressed activations with an effective precision of less than 1 bit and improved flexibility of precision allocation. In addition, we present a cost-effective yet accurate sensitivity calculation algorithm that solely relies on the L2 norm of parameter gradients, substantially reducing memory overhead due to sensitivity calculation. In experiments, the ALAM framework significantly reduces activation memory without compromising accuracy, achieving up to a 12.5x compression rate in LLMs. 

<p align="center">
  <img src="https://github.com/KH9NHAKRFF/ALAM/assets/144604248/884a3dad-861f-4948-98de-26316df644c8">
</p>

## Install

```bash
# install pytorch. we use torch 1.10.1 and torch 2.0.1, but other version is also possible 
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# build ALAM
pip install -v -e .
```

## Usage 

```python
import alam 
from alam.controller import Controller # import alam controller
# set the target bit and total_step during training. 
# alam will calculate the sensitivity at the initial and 10% of the total step. 
alam.set_config(bit=bit, total_step=total_step)  
model = .... # define your model here
controller = Controller(model, save_bit_path, load_bit_path) # define controller
# note that you can save allocated bits by setting save_bit_path
# and reuse it by setting load_bit_path for skipping allocating bits
# Once you obtain a bit_tensor, it can be used for fine-tuning other tasks as well,
# since empirically, the sensitivity distribution appears similar across tasks. 
def pack_hook(tensor): # quantize hook
    return controller.quantize(tensor)
        
def unpack_hook(tensor): # dequantize hook
    return controller.dequantize(tensor)

controller.install_hook()

with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
 for epoch in ...
   for iter in ....
     ......
     def backprop():
         model.train() # make sure you are in the training mode
         output = model(input) # forward
         loss = calculate_loss()
         optimizer.zero_grad() # this line must be present!
         loss.backward() # backward

     optimizer.step() # update the parameters
     controller.iterate(backprop) # tell gact how to perform forward/backward

controller.uninstall_hook()
```
## Results

### 1. Text classification
![results_text_classification](https://github.com/KH9NHAKRFF/ALAM/assets/144604248/f4b1d640-eb85-4611-a687-a2789de0fdbb)


### 2. Large Language Models
![results_llm](https://github.com/KH9NHAKRFF/ALAM/assets/144604248/e33d8fad-2380-4653-abb8-c131da9861f0)


## Example
[text_classification](https://github.com/KH9NHAKRFF/ALAM/tree/main/benchmark/text_classification)

[llm](https://github.com/KH9NHAKRFF/ALAM/tree/main/benchmark/llm)

 
## Acknowledgments
  
  In this repository, [GACT](https://github.com/LiuXiaoxuanPKU/GACT-ICML) are modified to develop our ALAM.
  Thanks the authors for open-source code.
  
 ## Lisense

> All content in this repository is licensed under the MIT license. 

