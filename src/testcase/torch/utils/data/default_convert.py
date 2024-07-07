import torch
import random
from collections import namedtuple
import numpy as np


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.utils.data.default_convert)
class TorchUtilsDataDefaultconvertTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_default_convert_correctness(self):
        # Generate random input data
        input_data_type = random.randint(0, 4)  # 0: int, 1: NumPy array, 2: NamedTuple with int, 3: NamedTuple with NumPy array, 4: List of NumPy array
    
        if input_data_type == 0:
            data = random.randint(-100, 100)  # Random integer
        elif input_data_type == 1:
            dim = random.randint(1, 4)  # Random dimension for the tensor
            num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
            input_size = [num_of_elements_each_dim for i in range(dim)]
            data = np.random.randn(*input_size)  # Random NumPy array
        elif input_data_type == 2:
            Point = namedtuple('Point', ['x', 'y'])
            data = Point(random.randint(-100, 100), random.randint(-100, 100))  # Random NamedTuple with int
        elif input_data_type == 3:
            Point = namedtuple('Point', ['x', 'y'])
            dim = random.randint(1, 4)  # Random dimension for the tensor
            num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
            input_size = [num_of_elements_each_dim for i in range(dim)]
            data = Point(np.random.randn(*input_size), np.random.randn(*input_size))  # Random NamedTuple with NumPy array
        elif input_data_type == 4:
            list_length = random.randint(1, 5)
            dim = random.randint(1, 4)  # Random dimension for the tensor
            num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
            input_size = [num_of_elements_each_dim for i in range(dim)]
            data = [np.random.randn(*input_size) for _ in range(list_length)]  # Random List of NumPy array
    
        result = torch.utils.data.default_convert(data)
        return result
    
    
    
    