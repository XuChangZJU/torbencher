import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.svd)
class TorchSvdTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_svd_correctness(self):
        # Randomly generate input tensor size
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim - 1)] + [random.randint(1, 5), random.randint(1,
                                                                                                                5)]  # The last two dimensions can be different

        input_tensor = torch.randn(input_size)
        u, s, v = torch.svd(input_tensor)
        return u, s, v
        # return torch.dist(input_tensor, torch.matmul(torch.matmul(u, torch.diag_embed(s)),
        #                                              v.mT))  # Calculate the distance between input and reconstructed matrix
