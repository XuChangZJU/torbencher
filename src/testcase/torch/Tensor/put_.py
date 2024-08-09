import torch
import random
from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.put_)
class TorchTensorPutUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_put__correctness(self):
        dim_of_tensor = random.randint(1, 3)
        size_of_dim = random.randint(1, 4)
        size = [size_of_dim for _ in range(dim_of_tensor)]

        origin = torch.randn(size)
        length = torch.numel(origin)

        num_to_put = random.randint(1, length)
        indices_to_put = torch.tensor([random.randint(0, num_to_put - 1) for _ in range(num_to_put)])
        val_to_put = torch.tensor([random.random() for _ in range(num_to_put)])

        origin.put_(indices_to_put, val_to_put)
        return origin
