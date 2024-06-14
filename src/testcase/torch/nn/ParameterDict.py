import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.ParameterDict)
class TorchNnParameterdictTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_parameterdict_correctness(self):
        # Random dimensions for the tensors
        dim1 = random.randint(1, 4)
        dim2 = random.randint(1, 4)
        
        # Random number of elements for each dimension
        num_elements_dim1 = random.randint(1, 5)
        num_elements_dim2 = random.randint(1, 5)
        
        # Random input sizes for the parameters
        input_size1 = [num_elements_dim1 for _ in range(dim1)]
        input_size2 = [num_elements_dim2 for _ in range(dim2)]
        
        # Creating random parameters
        param1 = torch.nn.Parameter(torch.randn(input_size1))
        param2 = torch.nn.Parameter(torch.randn(input_size2))
        
        # Creating a ParameterDict with random parameters
        param_dict = torch.nn.ParameterDict({
            'param1': param1,
            'param2': param2
        })
        
        # Randomly selecting a parameter from the ParameterDict
        choice = random.choice(['param1', 'param2'])
        
        # Creating a random input tensor for matrix multiplication
        input_tensor = torch.randn(param_dict[choice].size(-1), random.randint(1, 5))
        
        # Performing matrix multiplication using the selected parameter
        result = param_dict[choice].mm(input_tensor)
        
        return result
    