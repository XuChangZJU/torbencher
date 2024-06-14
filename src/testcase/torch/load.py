import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.load)
class TorchLoadTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_load_correctness(self):
    # Generate random data
    dim = random.randint(1, 4)
    num_of_elements_each_dim = random.randint(1, 5)
    input_size = [num_of_elements_each_dim for i in range(dim)]
    data = torch.randn(input_size)

    # Save the data to a file
    file_name = "test_tensor.pt"
    torch.save(data, file_name)

    # Load the data from the file
    loaded_data = torch.load(file_name)

    # Return the loaded data
    return loaded_data
