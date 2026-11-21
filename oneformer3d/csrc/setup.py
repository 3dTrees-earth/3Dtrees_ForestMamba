"""Build the fm_kernels CUDA extension (see Dockerfile.3dtrees).

    cd oneformer3d/csrc && TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6" python setup.py install
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="fm_kernels_cuda",
    version="1.0.0",
    ext_modules=[
        CUDAExtension(
            name="fm_kernels_cuda",
            sources=["fm_kernels.cu"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
