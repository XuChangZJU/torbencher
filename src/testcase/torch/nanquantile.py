import random
import torch


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nanquantile)
class TorchNanquantileTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_nanquantile_correctness(self):
        # Random dimension for the tensor (1 to 4)
        dim = random.randint(1, 4)
        # Random number of elements in each dimension (1 to 5)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]
    
        # Generate random tensor with some NaN values
        tensor = torch.randn(input_size)
        # Introducing some NaN values randomly
        num_of_nans = random.randint(0, num_of_elements_each_dim)
        for _ in range(num_of_nans):
            index = tuple(random.randint(0, num_of_elements_each_dim - 1) for _ in range(dim))
            tensor[index] = float('nan')
    
        # Random quantile value in the range [0, 1]
        quantile = random.uniform(0, 1)
        # Optional dimension reduction, None for no reduction
        dim = random.randint(0, len(input_size) - 1) if len(input_size) > 1 else None
    
        # Compute the nan quantile ignoring NaN values
        result = torch.nanquantile(tensor, quantile, dim)
        return result
    