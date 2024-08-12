import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.jit.ignore)
class TorchJitIgnoreTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_jit_ignore_correctness(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()

            @torch.jit.ignore
            def debugger(self, x):
                import pdb
                pdb.set_trace()

            def forward(self, x):
                x += 10
                self.debugger(x)
                return x

        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]  # Generate input size

        tensor_input = torch.randn(input_size)  # Generate random tensor input
        model = MyModule()  # Instantiate the model
        scripted_model = torch.jit.script(model)  # Script the model
        result = scripted_model(tensor_input)  # Run the model with the tensor input
        return result

    def test_jit_ignore_drop_correctness(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()

            @torch.jit.ignore(drop=True)
            def training_method(self, x):
                import pdb
                pdb.set_trace()

            def forward(self, x):
                if self.training:
                    self.training_method(x)
                return x

        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]  # Generate input size

        tensor_input = torch.randn(input_size)  # Generate random tensor input
        model = MyModule()  # Instantiate the model
        scripted_model = torch.jit.script(model)  # Script the model
        result = scripted_model(tensor_input)  # Run the model with the tensor input
        return result
