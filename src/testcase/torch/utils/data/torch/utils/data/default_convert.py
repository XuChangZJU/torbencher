import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.utils.data.torch.utils.data.default_convert)
class TorchUtilsDataTorchUtilsDataDefaultconvertTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_default_convert_correctness(self):
        # Randomly generate input data with different types
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        data = [
            random.randint(0, 100),  # int
            random.uniform(0.1, 10.0),  # float
            [random.randint(0, 100) for _ in range(random.randint(1, 5))],  # list of int
            [random.uniform(0.1, 10.0) for _ in range(random.randint(1, 5))],  # list of float
            (random.randint(0, 100), random.uniform(0.1, 10.0)),  # tuple
            torch.randn(input_size),  # tensor
        ]
        result = torch.utils.data.default_convert(data)
        return result
    
    
    
    