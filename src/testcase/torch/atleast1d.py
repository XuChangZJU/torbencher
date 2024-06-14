import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.atleast1d)
class TorchAtleast1dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_atleast_1d_correctness(self):
    # Random dimension for the first tensor
    dim1 = random.randint(0, 3)
    # If dim1 is zero, create as scalar, otherwise as a tensor with shape [dim1,...]
    tensor1 = torch.tensor(random.uniform(0.1, 10.0)) if dim1 == 0 else torch.randn([random.randint(1, 5) for _ in range(dim1)])
    
    # Random dimension for the second tensor (if we want to test with multiple tensors)
    dim2 = random.randint(0, 3)
    # If dim2 is zero, create as scalar, otherwise as a tensor with shape [dim2,...]
    tensor2 = torch.tensor(random.uniform(0.1, 10.0)) if dim2 == 0 else torch.randn([random.randint(1, 5) for _ in range(dim2)])
    
    if random.random() > 0.5:
        # Test case with a single tensor
        result = torch.atleast_1d(tensor1)
    else:
        # Test case with multiple tensors
        result = torch.atleast_1d((tensor1, tensor2))
    
    return result
