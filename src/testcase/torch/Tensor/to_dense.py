import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.to_dense)
class TorchTensorToUdenseTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_to_dense_correctness(self):
        # Randomly decide if the tensor will be sparse or dense
        is_sparse = random.choice([True, False])

        if is_sparse:
            # Create a random sparse tensor
            indices = torch.tensor([[random.randint(0, 2), random.randint(0, 2)],
                                    [random.randint(0, 2), random.randint(0, 2)]])
            values = torch.randn(indices.size(1))
            sparse_tensor = torch.sparse_coo_tensor(indices, values, size=(3, 3))
            result = sparse_tensor.to_dense()
        else:
            # Create a random dense tensor
            dense_tensor = torch.randn(3, 3)
            result = dense_tensor.to_dense()

        return result
