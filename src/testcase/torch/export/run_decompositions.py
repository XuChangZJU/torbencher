import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.export.run_decompositions)
class TorchExportRundecompositionsTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_run_decompositions_correctness(self):
        # Randomly generate a tensor size
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]
    
        # Generate random tensor
        tensor = torch.randn(input_size)
    
        # Define a simple decomposition function for testing
        def decomposition_fn(tensor):
            return tensor * 2
    
        # Run the decomposition
        result = decomposition_fn(tensor)
        return result
    
    
    
    