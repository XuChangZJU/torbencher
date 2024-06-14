import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.logit)
class TorchTensorLogitTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_logit__correctness(self):
    """
    Test the correctness of torch.Tensor.logit_()
    """
    dim = random.randint(1, 4)
    num_of_elements_each_dim = random.randint(1, 5)
    input_size = [num_of_elements_each_dim for i in range(dim)]

    # Generate random tensor data with values in (0, 1)
    input_tensor = torch.rand(input_size) 
    input_tensor.logit_() # In-place operation
    return input_tensor 
