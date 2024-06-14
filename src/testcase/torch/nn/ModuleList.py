import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.ModuleList)
class TorchNnModulelistTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_modulelist_correctness(self):
        # Define the number of submodules in the ModuleList
        num_submodules = random.randint(1, 5) 
    
        # Define the input size for the Linear modules
        input_size = random.randint(1, 10)
        output_size = random.randint(1, 10)
    
        # Create a list of Linear modules
        modules = [torch.nn.Linear(input_size, output_size) for _ in range(num_submodules)]
    
        # Create a ModuleList
        module_list = torch.nn.ModuleList(modules)
    
        # Generate random input tensor
        input_tensor = torch.randn(input_size)
    
        # Pass the input tensor through each submodule in the ModuleList
        results = []
        for module in module_list:
            results.append(module(input_tensor))
    
        # Return the list of results
        return results
    