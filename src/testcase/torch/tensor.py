import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.tensor)
class TorchTensorTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_tensor_correctness(self):
        # Generate random dimension for the tensor
        dim = random.randint(1, 4)
        # Generate random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Generate input size
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate random tensor data
        data = torch.randn(input_size)
        # Create tensor
        result = torch.tensor(data)
        # 测试生成整型张量
        int_data = [random.randint(0, 100) for _ in range(10)]
        int_tensor = torch.tensor(int_data, dtype=torch.int32)
        assert int_tensor.dtype == torch.int32, "Expected dtype to be torch.int32"
        assert int_tensor.shape == torch.Size([10]), "Expected shape to be [10]"
        assert (int_tensor == torch.tensor(int_data,
                                           dtype=torch.int32)).all(), "Tensor values do not match expected values"
        # 测试生成多维张量
        multi_dim_data = [[random.uniform(0, 100) for _ in range(3)] for _ in range(3)]
        multi_dim_tensor = torch.tensor(multi_dim_data, dtype=torch.float32)
        assert multi_dim_tensor.dtype == torch.float32, "Expected dtype to be torch.float32"
        assert multi_dim_tensor.shape == torch.Size([3, 3]), "Expected shape to be [3, 3]"
        assert torch.allclose(multi_dim_tensor, torch.tensor(multi_dim_data,
                                                             dtype=torch.float32)), "Tensor values do not match expected values"

        return result
