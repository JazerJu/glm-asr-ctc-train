from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="ctc_log_softmax_ext",
    ext_modules=[
        CUDAExtension(
            name="ctc_log_softmax_ext",
            sources=[
                "ctc_log_softmax.cpp",
                "ctc_log_softmax_cuda.cu",
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
