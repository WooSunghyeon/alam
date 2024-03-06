from setuptools import setup, find_packages
from torch.utils import cpp_extension

setup(name='alam',
      ext_modules=[
          cpp_extension.CUDAExtension(
              'alam.cpp_extension.calc_precision',
              ['alam/cpp_extension/calc_precision.cc']
          ),
          cpp_extension.CUDAExtension(
              'alam.cpp_extension.minimax',
              ['alam/cpp_extension/minimax.cc', 'alam/cpp_extension/minimax_cuda_kernel.cu']
          ),
          cpp_extension.CUDAExtension(
              'alam.cpp_extension.quantization',
              ['alam/cpp_extension/quantization.cc',
                  'alam/cpp_extension/quantization_cuda_kernel.cu']
          ),
      ],
      cmdclass={'build_ext': cpp_extension.BuildExtension},
      packages=find_packages()
      )
