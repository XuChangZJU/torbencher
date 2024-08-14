import random
import unittest

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.is_cuda)
class TorchTensorIsUcudaTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    @unittest.skipIf(not torch.cuda.is_available(),"CUDA is not available.")
    def test_is_cuda_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]  # Generate input size list

        tensor_cpu = torch.randn(input_size)  # Create a tensor on CPU
        tensor_gpu = torch.randn(input_size).cuda()  # Create a tensor on GPU

        result_cpu = tensor_cpu.is_cuda  # Check if the tensor is on GPU (should be False)
        result_gpu = tensor_gpu.is_cuda  # Check if the tensor is on GPU (should be True)

        return result_cpu, result_gpu
