from alam.conf import config
from alam.quantizer import Quantizer
from alam.autoprec import AutoPrecision
import torch

class Controller:
    def __init__(self, model, save_bit_path=None, load_bit_path=None):
        self.model = model
        
        assert(config.bit <= 16)
        
        if config.bit < 8:
            default_bit = 4
        elif config.bit < 16:
            default_bit = 8    
        else:
            assert(config.bit <= 16)
            default_bit = 8
        
        self.quantizer = Quantizer(
            default_bit=default_bit)
        # does not quantize model parameters
        self.quantizer.filter_tensors(model.named_parameters())
        self.ap = AutoPrecision(
            self.model, self.quantizer, config.bit, config.max_bit,
            config.work_dir, config.adapt_step, config.log_interval, save_bit_path=save_bit_path, load_bit_path=load_bit_path)

        self.bit = config.bit
        self.iter = 0

    def __del__(self):
        self.uninstall_hook()

    def iterate(self, get_grad):
        if not config.compress_activation:
            return
        self.quantizer.iterate()
        self.ap.iterate_wrapper(get_grad)
        self.iter += 1
        self.quantizer.seed_iter = self.iter

    def quantize(self, input):
        return self.quantizer.quantize(input)

    def dequantize(self, input):
        return self.quantizer.dequantize(input)
    
    def install_hook(self):
        def pack_hook(x):
            r = self.quantize(x)
            del x
            return r

        def unpack_hook(x):
            r = self.dequantize(x)
            del x
            return r

        if torch.__version__ < torch.torch_version.Version('1.10'):
            print("[Error] Please install PyTorch with version >= 1.10")
        elif torch.__version__ < torch.torch_version.Version('1.11'):
            torch._C._autograd._register_saved_tensors_default_hooks(
                pack_hook, unpack_hook)
        else:
            torch._C._autograd._push_saved_tensors_default_hooks(
                pack_hook, unpack_hook)

    def uninstall_hook(self):
        if torch.__version__ < torch.torch_version.Version('1.10'):
            print("[Error] Please install PyTorch with version >= 1.10")
        elif torch.__version__ < torch.torch_version.Version('1.11'):
            torch._C._autograd._reset_saved_tensors_default_hooks()
        else:
            torch._C._autograd._pop_saved_tensors_default_hooks()
