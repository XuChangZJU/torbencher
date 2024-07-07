import torch
import random
from collections import namedtuple


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.utils.data.default_collate)
class TorchUtilsDataDefaultcollateTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_default_collate_tensor(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1,5) # Random number of elements each dimension
        input_size=[num_of_elements_each_dim for i in range(dim)] 
        batch = [torch.randn(input_size) for _ in range(random.randint(1, 10))]  # Generate a batch of tensors with random size
        result = torch.utils.data.default_collate(batch)
        return result
    
    def test_default_collate_numpy_arrays(self):
        dim = random.randint(1, 4)  # Random dimension for the arrays
        num_of_elements_each_dim = random.randint(1,5) # Random number of elements each dimension
        input_size=[num_of_elements_each_dim for i in range(dim)] 
        batch = [torch.randn(input_size).numpy() for _ in range(random.randint(1, 10))]  # Generate a batch of numpy arrays with random size
        result = torch.utils.data.default_collate(batch)
        return result
    
    def test_default_collate_float(self):
        batch = [random.uniform(0.1, 10.0) for _ in range(random.randint(1, 10))]  # Generate a batch of floats
        result = torch.utils.data.default_collate(batch)
        return result
    
    def test_default_collate_int(self):
        batch = [random.randint(1, 100) for _ in range(random.randint(1, 10))]  # Generate a batch of integers
        result = torch.utils.data.default_collate(batch)
        return result
    
    def test_default_collate_str(self):
        batch = ['a', 'b', 'c']  # Generate a batch of strings
        result = torch.utils.data.default_collate(batch)
        return result
    
    def test_default_collate_bytes(self):
        batch = [b'a', b'b', b'c']  # Generate a batch of bytes
        result = torch.utils.data.default_collate(batch)
        return result
    
    def test_default_collate_mapping(self):
        batch = [{'A': random.randint(0, 100), 'B': random.randint(0, 100)} for _ in range(random.randint(1, 10))]  # Generate a batch of dictionaries
        result = torch.utils.data.default_collate(batch)
        return result
    
    def test_default_collate_namedtuple(self):
        Point = namedtuple('Point', ['x', 'y'])
        batch = [Point(random.randint(0, 100), random.randint(0, 100)) for _ in range(random.randint(1, 10))]  # Generate a batch of namedtuples
        result = torch.utils.data.default_collate(batch)
        return result
    
    def test_default_collate_tuple(self):
        batch = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(random.randint(1, 10))]  # Generate a batch of tuples
        result = torch.utils.data.default_collate(batch)
        return result
    
    def test_default_collate_list(self):
        batch = [[random.randint(0, 100), random.randint(0, 100)] for _ in range(random.randint(1, 10))]  # Generate a batch of lists
        result = torch.utils.data.default_collate(batch)
        return result
    
    
    
    
    
    
    
    
    
    
    
    
    