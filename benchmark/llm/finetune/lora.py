"""
Instruction-tuning with LoRA on the Alpaca dataset.

Note: If you run into a CUDA error "Expected is_sm80 to be true, but got false", uncomment the line
`torch.backends.cuda.enable_flash_sdp(False)` in the script below (see https://github.com/Lightning-AI/lit-llama/issues/101).
"""
import sys
from pathlib import Path
import os
import time

import lightning as L
import numpy as np
import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate import generate
from lit_llama.lora import mark_only_lora_as_trainable, lora, lora_state_dict
from lit_llama.model import LLaMA, LLaMAConfig
from lit_llama.tokenizer import Tokenizer
from scripts.prepare_alpaca import generate_prompt
from lightning.fabric.plugins import BitsandbytesPrecision
import bitsandbytes as bnb

import alam
from alam.controller import Controller

torch.backends.cuda.enable_flash_sdp(False)

instruction_tuning = True
eval_interval = 35
save_interval = 35
eval_iters = 100
log_interval = 1
devices=1

# Hyperparameters
learning_rate = 1e-4
batch_size = 128
micro_batch_size = 2
gradient_accumulation_iters = batch_size // micro_batch_size // devices
assert gradient_accumulation_iters > 0
epoch_size = 50000
num_epochs = 1
max_iters = num_epochs * (epoch_size // micro_batch_size) // devices
weight_decay = 0.0
max_seq_length = 512  # see scripts/prepare_alpaca.py
lora_r = 8
lora_alpha = 16
lora_dropout = 0.05
warmup_iters = int(0.1 * max_iters // gradient_accumulation_iters)  # 1 epoch

def main(
    data_dir: str = "data/alpaca", 
    pretrained_path: str = "checkpoints/lit-llama/7B/lit-llama.pth",
    tokenizer_path: str = "checkpoints/lit-llama/tokenizer.model",
    out_dir: str = "out/lora/alpaca",
    qlora: bool = False,
    do_alam: bool = True,
    bit: int = 2,
    save_bit_dir: str = None,
    load_bit_dir: str = None,
):
    
    alam.set_config(bit=bit, total_step=max_iters)
    if qlora:
        plugins = BitsandbytesPrecision("nf4", torch.bfloat16)
        fabric = L.Fabric(accelerator="cuda", strategy="auto", devices=devices, precision=None, plugins=plugins)
    else:
        fabric = L.Fabric(accelerator="cuda", devices=devices, precision="bf16-true")
    
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    train_data, val_data = load_datasets(data_dir=data_dir)

    config = LLaMAConfig.from_name("7B")
    config.block_size = max_seq_length
    checkpoint = torch.load(pretrained_path)
    new_checkpoint = {}
    for name, param in checkpoint.items():
        # Split the name and insert '.linear' where needed
        name_parts = name.split('.')
        if 'c_attn' in name_parts:
            # Find the index of 'c_attn' and insert 'linear' after it
            c_attn_index = name_parts.index('c_attn')
            name_parts.insert(c_attn_index + 1, 'linear')
        
        # Rejoin the modified name parts
        new_name = '.'.join(name_parts)
        new_checkpoint[new_name] = param
    with fabric.init_module(), lora(r=lora_r, alpha=lora_alpha, dropout=lora_dropout, enabled=True):
        model = LLaMA(config)
        
    mark_only_lora_as_trainable(model)

    if qlora:
        optimizer = bnb.optim.PagedAdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters// gradient_accumulation_iters - warmup_iters)
    model, optimizer = fabric.setup(model, optimizer)
    model.load_state_dict(new_checkpoint, strict=False)
    train(fabric, model, optimizer, scheduler, train_data, val_data, tokenizer_path, out_dir, do_alam, save_bit_dir, load_bit_dir)

    # Save the final LoRA checkpoint at the end of training
    checkpoint = lora_state_dict(model)
    fabric.save(os.path.join(out_dir, "lit-llama-lora-finetuned.pth"), checkpoint)


def train(
    fabric: L.Fabric,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    train_data: np.ndarray,
    val_data: np.ndarray,
    tokenizer_path: str,
    out_dir: str,
    do_alam: bool,
    save_bit_dir: str,
    load_bit_dir: str,
) -> None:
    """The training loop.

    Loosely based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
    """
    step_count = 0
    train_data_list = []  # List to store [iter_num, dt, loss]
    val_data_list = []  # List to store [iter_num, val_loss]
    
    save_bit_path=None
    load_bit_path=None
    if save_bit_dir != None:
        save_bit_path = os.path.join(save_bit_dir, "bit_tensor.pt")
    if load_bit_dir != None:
        load_bit_path = os.path.join(load_bit_dir, "bit_tensor.pt")
    
    controller = Controller(model, save_bit_path, load_bit_path)
    def pack_hook(tensor): # quantize hook
        if not(do_alam):
            return tensor
        return controller.quantize(tensor)
    def unpack_hook(tensor): # dequantize hook
        if not(do_alam):
            return tensor
        return controller.dequantize(tensor)
    
    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        for iter_num in range(max_iters):

            if step_count <= warmup_iters:
                # linear warmup
                lr = learning_rate * step_count / warmup_iters
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            t0 = time.time()

            input_ids, targets = get_batch(fabric, train_data)
            with fabric.no_backward_sync(model, enabled=((iter_num + 1) % gradient_accumulation_iters != 0)):
                logits = model(input_ids)
                loss = loss_fn(logits, targets)
                fabric.backward(loss / gradient_accumulation_iters)
                

            torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 0.5)

            if (iter_num + 1) % gradient_accumulation_iters == 0:
                optimizer.step()
                optimizer.zero_grad()
                if step_count > warmup_iters:
                    scheduler.step()
                step_count += 1
                    
                if step_count % eval_interval == 0:
                    val_loss = validate(fabric, model, val_data, tokenizer_path)
                    fabric.print(f"step {iter_num}: val loss {val_loss:.4f}")
                    fabric.barrier()
                    val_data_list.append([iter_num, val_loss])
                    val_data_arr = np.array(val_data_list)
                    np.save(os.path.join(out_dir, "val_data.txt"), val_data_arr)

                if step_count % save_interval == 0:
                    print(f"Saving LoRA weights to {out_dir}")
                    # We are only saving the LoRA weights
                    # TODO: Provide a function/script to merge the LoRA weights with pretrained weights
                    checkpoint = lora_state_dict(model)
                    fabric.save(os.path.join(out_dir, f"iter-{iter_num:06d}-ckpt.pth"), checkpoint)

            if iter_num == 0:
                optimizer.zero_grad()

            dt = time.time() - t0

            def backprop():
                with fabric.no_backward_sync(model, enabled=False):
                    model.train() # make sure you are in the training mode
                    logits = model(input_ids[:1]) # forward
                    loss = loss_fn(logits, targets[:1])
                    optimizer.zero_grad() # this line must be present!
                    loss.backward() # backward
            
            if alam:
                controller.iterate(backprop)

            train_data_list.append([iter_num, dt * 1000, loss.item()])

            if iter_num % log_interval == 0:
                fabric.print(f"iter {iter_num}: loss {loss.item():.4f}, time: {dt*1000:.2f}ms")

            # Save train data and val data after all iterations
            if iter_num % 10 == 0:
                train_data_arr = np.array(train_data_list)
                np.save(os.path.join(out_dir, "train_data.txt"), train_data_arr)

        val_loss = validate(fabric, model, val_data, tokenizer_path)
        
def generate_response(model, instruction, tokenizer_path):
    tokenizer = Tokenizer(tokenizer_path)
    sample = {"instruction": instruction, "input": ""}
    prompt = instruction
    if instruction_tuning:
        prompt = generate_prompt(sample)
    encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device)

    output = generate(
        model,
        idx=encoded,
        max_seq_length=max_seq_length,
        max_new_tokens=100,
    )
    output = tokenizer.decode(output)
    return output # output.split("### Response:")[1].strip()


@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_data: np.ndarray, tokenizer_path: str) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        input_ids, targets = get_batch(fabric, val_data)
        logits = model(input_ids)
        loss = loss_fn(logits, targets)
        losses[k] = loss.item()
    out = losses.mean()

    # produce an example:
    instruction = "Recommend a movie for me to watch during the weekend and explain the reason."
    
    output = generate_response(model, instruction, tokenizer_path)
    fabric.print(instruction)
    fabric.print(output)

    model.train()
    return out.item()

def loss_fn(logits, targets):
    # shift the targets such that output n predicts token n+1
    logits = logits[..., :-1, :].contiguous()
    targets = targets[..., 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
    return loss
    

def get_batch(fabric: L.Fabric, data: list):
    ix = torch.randint(len(data), (micro_batch_size,))

    input_ids = [data[i]["input_ids"].type(torch.int64) for i in ix]
    labels = [data[i]["labels"].type(torch.int64) for i in ix]

    max_len = max(len(s) for s in input_ids)

    def pad_right(x, pad_id):
        # pad right based on the longest sequence
        n = max_len - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
    y = torch.stack([pad_right(x, pad_id=-1) for x in labels])
    x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))
    return x, y


def load_datasets(data_dir):
    train_data = torch.load(os.path.join(data_dir, "train.pt"))
    val_data = torch.load(os.path.join(data_dir, "test.pt"))
    return train_data, val_data


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")
    
    from jsonargparse.cli import CLI

    CLI(main)
