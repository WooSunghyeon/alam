import math
import numpy as np
import torch
import torch.nn as nn

from alam.conf import config
import alam.cpp_extension.quantization as ext_quantization
import alam.cpp_extension.minimax as ext_minimax
from torch.utils.checkpoint import checkpoint

def no_scheme_quantize_pack(input, q_bit, seed):
    group_size = config.group_size
    N = (input.numel() + group_size - 1) //  group_size
    num_ele = N * group_size
    pad_num = num_ele - input.numel()
    if pad_num > 0:
        input = torch.cat([input.reshape(1, -1), torch.zeros([1, pad_num], 
                                            dtype=input.dtype, device=input.device)], dim=1)
    input_groups = input.reshape(-1, group_size)  
    if q_bit == 32:  # TODO, use kernel to optimize this
        q_min = input_groups.min(dim=-1, keepdim=True).values
        q_scale = input_groups.max(dim=-1, keepdim=True).values - q_min
        q_input = input_groups
    else:
        q_min, mx = ext_minimax.minimax(input_groups)
        input_groups = input_groups.view(N, -1, group_size)
        q_min = q_min.view(N, -1, 1)
        mx = mx.view(N, -1, 1)
        q_input, q_scale = ext_quantization.pack_single_precision(input_groups.contiguous(), q_min, mx, q_bit, True, seed)
        del mx
    return q_input, q_scale, q_min, pad_num

def dequantize_and_unpack(data, shape, q_bit, scale, mn):
    group_size = config.group_size
    if not isinstance(q_bit, int):
        print("bits must be intergers, now bits ", q_bit)
        assert(False)
        
    if q_bit == 32:
        return data    
    unpack_func = ext_quantization.unpack_single_precision
    num_groups = (int(np.prod(shape)) + group_size - 1)  // group_size
    return unpack_func(
        data, q_bit, scale, mn, num_groups, 1, group_size
    )


def op_quantize_mask(input):
    return [ext_quantization.act_quantize_dropout_mask(input.contiguous()), input.shape]

def op_dequantize_mask(input):
    q_mask, input_shape = input
    output = ext_quantization.act_dequantize_dropout_mask(q_mask, np.prod(input_shape)).view(input_shape)
    return output

def op_quantize(input, q_bit, seed):
    if input.dtype == torch.bfloat16:
        input = input.float()
    delta=0
    # when AQ 0.5-bit is allocated. Note that q_bit=1 represents AQ 0.5-bit, not conventional 1-bit.
    if q_bit == 1:
        remainder = input.numel() % config.average_group_size
        # extract group average of activations
        if remainder == 0:
            input = input.reshape(-1,config.average_group_size)
            input = input.mean(dim=1, keepdim=True)
        else:
            input_set = input.view(1,-1)[:,:input.numel()-remainder]
            input_remainder = input.view(1,-1)[:,input.numel()-remainder:]
            input_set = input_set.view(-1,config.average_group_size).mean(dim=1, keepdim=True)
            input_remainder = input_remainder.mean(dim=1,keepdim=True)
            input = torch.cat([input_set,input_remainder],dim=0)
        # quantize group value by 2-bit
        q_input_shape = input.shape
        q_input, q_scale, q_min, pad_num = no_scheme_quantize_pack(input, int(2*q_bit), seed)
        return [q_input, q_bit, q_scale, q_min, q_input_shape, remainder, pad_num] 
    # when >2 bit is allocated, only conventional quantization is performed
    else:
        q_input_shape = remainder = None
        q_input, q_scale, q_min, pad_num = no_scheme_quantize_pack(input, q_bit, seed)
        return [q_input, q_bit, q_scale, q_min, q_input_shape, remainder, pad_num]
        
def op_dequantize(input, input_shape):
    q_input, q_bit, q_scale, q_min, q_input_shape, remainder, pad_num = input
    # when AQ 0.5-bit is allocated. Note that q_bit=1 represents AQ 0.5-bit, not conventional 1-bit.
    if q_bit == 1:
        # perfor 2-bit dequantization first
        input = dequantize_and_unpack(
            q_input, q_input_shape, int(2*q_bit), q_scale, q_min)
        if pad_num != 0:
            input = input.ravel()[:-pad_num]
        input = input.view(q_input_shape)
        # perfor recover activations by repeating decompressed group average
        if remainder == 0:
            input = input.repeat(1, config.average_group_size).view(input_shape)
        else:
            set = np.prod(input_shape)//config.average_group_size
            input_set = input[:set,:].repeat(1,config.average_group_size).view(-1,1)
            input_remainder = input[set:,:].repeat(1,remainder).view(-1,1)
            input = torch.cat([input_set,input_remainder],dim=0).view(input_shape)
        num_features = np.prod(input_shape)
        input = input.ravel()[:num_features]
        input = input.view(*input_shape).contiguous()
        return input
    # when >2 bit is allocated, only conventional dequantization is performed
    else:
        input = dequantize_and_unpack(
            q_input, input_shape, q_bit, q_scale, q_min)
        num_features = np.prod(input_shape)
        input = input.ravel()[:num_features]
        input = input.view(*input_shape).contiguous()
        return input        
 
            
# Implementation of efficient self attention
# https://arxiv.org/abs/2112.05682
def self_atten(dropout_p, query_layer, key_layer, value_layer,
               q_chunk_size, k_chunk_size, use_checkpoint=True):
    batch_size, num_heads, seq_len, q_features = query_layer.shape
    batch_size, num_heads, seq_len, k_features = key_layer.shape
    batch_size, num_heads, seq_len, v_features = value_layer.shape
    q_chunk_size = min(q_chunk_size, seq_len)
    dropout = nn.Dropout(dropout_p)

    def _query_chunk_attention(query, key, value):
        batch_size, num_heads, num_kv, k_features = key.shape
        v_features = value.shape[-1]
        key_chunk_size = min(k_chunk_size, num_kv)
        num_key_chunk = math.ceil(num_kv / key_chunk_size)
        query = query / math.sqrt(k_features)

        def summarize_chunk(query, key, value):
            attn_weights = torch.einsum('bhqd,bhkd->bhqk', query, key)
            max_score = torch.max(attn_weights, axis=-1, keepdims=True).values
            max_score = max_score.detach()
            exp_weights = torch.exp(attn_weights - max_score)
            exp_values = torch.einsum('bhvf,bhqv->bhqf', value, exp_weights)
            exp_values = dropout(exp_values)
            return (exp_values, exp_weights.sum(axis=-1), max_score.squeeze())

        chunk_values = None
        chunk_weights = None
        global_max = None

        def batch_dot(m1, m2):
            feature_size = m1.shape[-1]
            v = m1.view(-1, feature_size) * m2.view(-1, 1)
            return v.view(m1.shape)

        for i in range(num_key_chunk):
            key_chunk = key[:, :, i *
                            key_chunk_size: (i+1) * key_chunk_size, :]
            value_chunk = value[:, :, i *
                                key_chunk_size: (i+1) * key_chunk_size, :]
            if use_checkpoint:
                chunk_value, chunk_weight, chunk_max = \
                    checkpoint(
                        summarize_chunk, query, key_chunk, value_chunk)
            else:
                chunk_value, chunk_weight, chunk_max = summarize_chunk(
                    query, key_chunk, value_chunk)

            if global_max is None:
                global_max = chunk_max
                chunk_values = chunk_value
                chunk_weights = chunk_weight
            else:
                old_max = global_max
                global_max = torch.maximum(chunk_max, global_max).detach()

                diff1 = torch.exp(chunk_max - global_max).detach()
                chunk_value = batch_dot(chunk_value, diff1)
                chunk_weight *= diff1

                diff2 = torch.exp(old_max - global_max).detach()
                chunk_values = batch_dot(chunk_values, diff2)
                chunk_weights *= diff2

                chunk_values += chunk_value
                chunk_weights += chunk_weight

        chunk_values = chunk_values.view(-1, chunk_values.shape[-1])
        chunk_weights = chunk_weights.view(-1, 1)
        return chunk_values / chunk_weights

    num_q_chunk = math.ceil(query_layer.shape[2] / q_chunk_size)
    res = torch.zeros(query_layer.shape).cuda()
    for i in range(num_q_chunk):
        r = _query_chunk_attention(query_layer[:, :, i*q_chunk_size:(i+1)*q_chunk_size, :],
                                   key_layer, value_layer)
        res[:, :, i*q_chunk_size:(i+1)*q_chunk_size, :] = r.view(
            batch_size, num_heads, q_chunk_size, q_features)
    return res

#%%



