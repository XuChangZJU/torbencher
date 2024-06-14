import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.addmv)
class TorchTensorAddmvTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_addmv_correctness(self):
    # Random dimensions for the matrix and vector
    rows = random.randint(1, 5)
    cols = random.randint(1, 5)
    
    # Randomly generate matrix and vector with appropriate sizes
    mat = torch.randn(rows, cols)
    vec = torch.randn(cols)
    
    # Randomly generate beta and alpha values
    beta = random.uniform(0.1, 10.0)
    alpha = random.uniform(0.1, 10.0)
    
    # Randomly generate a tensor for the result with appropriate size
    result_tensor = torch.randn(rows)
    
    # Perform the addmv operation
    result = result_tensor.addmv(mat, vec, beta=beta, alpha=alpha)
    return result
