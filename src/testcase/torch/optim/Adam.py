import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.optim.Adam)
class TorchOptimAdamTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_adam_correctness(self):
    # Random dimension for the tensors
    dim = random.randint(1, 4)
    # Random number of elements each dimension
    num_of_elements_each_dim = random.randint(1, 5)
    input_size = [num_of_elements_each_dim for i in range(dim)]

    # Random tensor data
    tensor_data = torch.randn(input_size, requires_grad=True)

    # Construct optimizer
    optimizer = torch.optim.Adam([tensor_data])

    # Random learning rate
    lr = random.uniform(0.01, 0.1)

    # Perform optimization step
    for i in range(random.randint(1, 10)):
        # Random gradient
        gradient = torch.randn(input_size)
        # Update tensor data with gradient
        tensor_data.grad = gradient
        # Perform optimization step
        optimizer.step()
        # Zero the gradients
        optimizer.zero_grad()

    return tensor_data
