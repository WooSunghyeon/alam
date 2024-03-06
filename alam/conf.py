def set_config(bit=2, total_step=10000):
    config.bit=bit
    config.adapt_step = int(total_step*0.1)
    
class QuantizationConfig:
    def __init__(self):
        # compress activation, this field is set to False when we do not compress activation
        self.compress_activation = True        
        # average number of bits for activation
        # if auto precision is turned on, each activation is quantized uniformly with self.bit bits
        # perform group-wise quantization
        self.bit = 2
        self.group_size = 128 
        # average group_size, and then 2-bit quantization: AQ 0.5-bit
        self.average_group_size = 4 
        self.aq_bit = 0.5
        # avoid the same activation multiple times, this will further reduce training memory
        self.check_dup = True
        # the interval to adapt activation sensitivity
        self.adapt_step = 10000  
        # max number of bits for quantization
        self.max_bit = 32
        # it skips calculating sensitivity (set the sensitivity to random values), which enables fast memory check.
        self.fast_mem_check = False
        # set log_interval = -1 to disable logging
        # log debug information under the self.work_dir directory
        self.log_interval = -1 
        self.work_dir = "./log/"  
        
config = QuantizationConfig()
