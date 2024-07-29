import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.compile)
class TorchCompileTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_compile_correctness(self):
        dim = random.randint(1, 3)  # Random dimension for the tensor
        num_of_elements_each_dim = random.randint(1, 3)  # Random number of elements in each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]  # Create the tensor shape

        def model_function(x):
            return torch.sin(x) + torch.cos(x)  # Simple function to test torch.compile

        x = torch.randn(input_size)  # Generate random tensor for input

        # Possible modes for testing
        modes = ["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"]
        mode = random.choice(modes)  # Randomly choose a mode

        compiled_model = torch.compile(model_function, mode=mode)  # Compile the model function with a chosen mode
        result = compiled_model(x)  # Execute the compiled model
        return result
