import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.tensorsplit)
class TorchTensorTensorsplitTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_tensor_split_correctness(self):
        # Random dimension for the tensor
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]
        
        # Generate a random tensor
        tensor = torch.randn(input_size)
        
        # Randomly choose between indices or sections
        if random.choice([True, False]):
            # Randomly generate indices for splitting
            indices = sorted(random.sample(range(1, num_of_elements_each_dim), random.randint(1, num_of_elements_each_dim - 1)))
            result = tensor.tensor_split(indices, dim=0)
        else:
            # Randomly generate number of sections for splitting
            sections = random.randint(1, num_of_elements_each_dim)
            result = tensor.tensor_split(sections, dim=0)
        
        return result
    