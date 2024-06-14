import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.utils.vectortoparameters)
class TorchNnUtilsVectortoparametersTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_vector_to_parameters_correctness(self):
    # Randomly generate the number of parameters
    num_params = random.randint(1, 5)
    
    # Randomly generate the size of each parameter tensor
    param_sizes = [tuple(random.randint(1, 4) for _ in range(random.randint(1, 3))) for _ in range(num_params)]
    
    # Create a list of parameter tensors with the generated sizes
    parameters = [torch.randn(size) for size in param_sizes]
    
    # Flatten all parameter tensors into a single vector
    vec = torch.cat([p.view(-1) for p in parameters])
    
    # Apply vector_to_parameters to copy slices of vec into parameters
    torch.nn.utils.vector_to_parameters(vec, parameters)
    
    # Return the modified parameters to check correctness
    return parameters
