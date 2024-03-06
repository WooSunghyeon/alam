import torch
from alam.conf import config
from alam.ops import op_quantize, op_dequantize, op_quantize_mask, op_dequantize_mask
from alam.utils import uniform_sample


class Quantizer:
    """
    default_bit: the number of bits used to quantize
    """
    def __init__(self, default_bit):
        self.unrelated_tensors = set()  # record the tensors that should not be quantized
        self.default_bit = default_bit

        self.compute_stream = torch.cuda.current_stream()
        self.ptr_qtensor_map = {}
        self.layer_key_map = {}
        self.tid = 0
        self.start_bwd = True

        # data collected for auto precision
        self.seeds = {}
        self.bits = {}
        self.dims = {}

        self.iter = 0  # total number of iterations, including the extra inter for auto precision
        # iteration for seed, share the same seed_iter for the same auto precision adaptive step
        self.seed_iter = 0

    def filter_tensors(self, pairs):
        for _, v in pairs:
            self.unrelated_tensors.add(v.data_ptr())

    # return should_be_quantized, is_dropout_mask
    # treat dropout mask differently because it can be quantized with 1 bit with a specialized kernel
    def check_quantize(self, input_tensor, tid=1):
        # does not quantize parameters
        if input_tensor.data_ptr() in self.unrelated_tensors:
            return False, False
        # special check for saved mask
        if input_tensor.numel() > 0 and input_tensor.dtype == torch.uint8:
            if (input_tensor.max() == 1) and (input_tensor.min() == 0):
                return True, True
            return False, False
        # only quantize float16 and float32 amd bfloat16
        if input_tensor.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            return False, False
        # only quantize activation that requires gradient
        # for example: BN statistics (running mean/var) should not be quantized
        if input_tensor.requires_grad is False:
            return False, False
        # only quantize 2/3/4D tensors for now
        if ((len(input_tensor.shape) != 2)
            and (len(input_tensor.shape) != 3)
            and (len(input_tensor.shape) != 4)
            ):
            return False, False
        return True, False

    def __del__(self):
        del self.ptr_qtensor_map
        del self.layer_key_map
        del self.unrelated_tensors

    def iterate(self):
        del self.ptr_qtensor_map
        del self.layer_key_map
        self.ptr_qtensor_map = {}
        self.layer_key_map = {}
        self.tid = 0
        self.start_bwd = True
        self.iter += 1

    def generate_tensor_key(self, t, tid):
        if config.check_dup:
            # sample 100 elements data pointer + tensor.sum() as the key
            sample_cnt = min(100, t.numel())
            key = uniform_sample(t, sample_cnt, add_dataptr=True)
            key.append(t.sum().item())
            return tuple(key)
        else:
            return (tid)

    def quantize(self, input):
        origtype= input.dtype
        quantize, is_dropout_mask = self.check_quantize(input)
        if not quantize:
            return False, input, origtype
        # special case: use 1 bit to quantize dropout mask
        if is_dropout_mask:
            q_inputs = op_quantize_mask(input)
            return True, is_dropout_mask, q_inputs, origtype
        tid = self.tid
        self.tid += 1
        input_shape = input.shape
        key = self.generate_tensor_key(input, tid)
        self.layer_key_map[tid] = key
        skip_quantize = key in self.ptr_qtensor_map
        if not skip_quantize:
            if self.iter == 0:
                bit = self.default_bit
                self.bits[tid] = bit
                self.dims[tid] = input.numel()
                self.seeds[tid] = tid
            else:
                bit = self.bits[tid]
            # quantize
            q_inputs = op_quantize(
                input, bit, self.seeds[tid] + self.seed_iter)
            self.ptr_qtensor_map[key] = [q_inputs, 1, tid]
        else:
            # increase the ref count
            self.ptr_qtensor_map[key][1] += 1
            
        return True, is_dropout_mask, key, input_shape, tid, origtype

    def dequantize(self, input):
        quantized = input[0]
        if not quantized:
            if input[1].dtype != input[2]:
                input = input.to(input[2])
            return input[1]
        
        is_dropout_mask = input[1]
        if is_dropout_mask:
            _, is_dropout_mask, q_inputs, origtype = input
            ret = op_dequantize_mask(q_inputs)
            if ret.dtype != origtype:
                ret=ret.to(origtype) 
            return ret
        _, _, key, input_shape, tid, origtype = input
        q_inputs, ref_cnt, key_tid = self.ptr_qtensor_map[key]

        if not q_inputs[0].is_cuda:
            q_inputs[0] = q_inputs[0].cuda(non_blocking=False)

        ret = op_dequantize(q_inputs, input_shape)
        if ret.dtype != origtype:
            ret=ret.to(origtype)
        ref_cnt -= 1
        if ref_cnt < 0:
            print("[Error] Ref count < 0", key, ref_cnt)
            exit(-1)
        elif ref_cnt == 0:
            del self.ptr_qtensor_map[key]
        else:
            self.ptr_qtensor_map[key] = [q_inputs, ref_cnt, key_tid]
        return ret
