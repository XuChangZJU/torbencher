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

        # Reshape the selected parameter to ensure it's 2D
        param_reshaped = param_dict[choice].reshape(-1, param_dict[choice].size(-1))

        # Correctly define input_tensor to have compatible shape for multiplication
        # Here, we assume param_reshaped's second dimension (columns) should match input_tensor's first dimension (rows)
        input_tensor = torch.randn(param_reshaped.size(1), random.randint(1, 5))  # Match columns of param_reshaped

        # Perform matrix multiplication
        result = torch.matmul(param_reshaped, input_tensor)

        return result
