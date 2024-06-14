import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.LazyBatchNorm1d)
class TorchNnLazybatchnorm1dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_lazy_batch_norm_1d_correctness(self):
    # Random dimension for the tensor (batch size, num_features, length)
    batch_size = random.randint(1, 4)
    num_features = random.randint(1, 5)
    length = random.randint(1, 10)
    
    # Random input tensor
    input_tensor = torch.randn(batch_size, num_features, length)
    
    # Create LazyBatchNorm1d layer
    lazy_batch_norm = torch.nn.LazyBatchNorm1d()
    
    # Apply LazyBatchNorm1d to the input tensor
    result = lazy_batch_norm(input_tensor)
    
    return result
