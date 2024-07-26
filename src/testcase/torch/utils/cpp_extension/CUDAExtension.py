import os
import torch
import random
from torch.utils.cpp_extension import CUDAExtension

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.utils.cpp_extension.CUDAExtension)
class TorchUtilsCppextensionCudaextensionTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cuda_extension_correctness(self):
        # Ensure CUDA_HOME environment variable is set
        if 'CUDA_HOME' not in os.environ:
            raise EnvironmentError(
                "CUDA_HOME environment variable is not set. Please set it to your CUDA install root.")

        # Randomly generate the name of the extension
        extension_name = f"extension_{random.randint(1, 1000)}"

        # Randomly generate the source files list
        num_sources = random.randint(1, 3)
        sources = [f"source_{i}.cu" for i in range(num_sources)]

        # Create the CUDAExtension object
        cuda_extension = CUDAExtension(extension_name, sources)

        # Return the created CUDAExtension object to verify correctness
        return cuda_extension
