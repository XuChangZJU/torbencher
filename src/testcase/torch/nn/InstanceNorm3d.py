import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.InstanceNorm3d)
class TorchNnInstancenorm3dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_InstanceNorm3d_correctness(self):
        # Random input size
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # num_features should be equal to one of the dimensions of input tensor
        num_features = input_size[random.randint(0, len(input_size) - 1)]
        input_size = [20, num_features, 35, 45, 10]
    
        input_tensor = torch.randn(input_size)
        instance_norm_3d = torch.nn.InstanceNorm3d(num_features)
        result = instance_norm_3d(input_tensor)
        return result
    