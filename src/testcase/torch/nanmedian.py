import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nanmedian)
class TorchNanmedianTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_nanmedian_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_elements_each_dim for _ in range(dim)]
    
        input_tensor = torch.randn(input_size)
    
        # Introducing NaN values randomly into the tensor
        nan_count = random.randint(0, num_elements_each_dim)  # Random number of NaNs to introduce
        flattened_tensor = input_tensor.flatten()
        for _ in range(nan_count):
            flattened_tensor[random.randint(0, flattened_tensor.size(0) - 1)] = float('nan')
        input_tensor = flattened_tensor.view(input_size)
    
        result = torch.nanmedian(input_tensor)
        return result
    