import torch
import random
import array


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.frombuffer)
class TorchFrombufferTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_frombuffer_correctness(self):
        # Generate a random type size, choose from float32 (4 bytes) or int32 (4 bytes)
        dtype = random.choice([torch.float32, torch.int32])
        dtype_size = 4  # Both chosen types are 4 bytes
    
        # Generate a random number of elements in the buffer (between 1 and 10 elements)
        num_elements = random.randint(1, 10)
    
        # Create a buffer with random data
        if dtype == torch.float32:
            # Buffer of random floats
            buffer = array.array('f', [random.uniform(-10, 10) for _ in range(num_elements)])
        else:
            # Buffer of random ints
            buffer = array.array('i', [random.randint(-10, 10) for _ in range(num_elements)])
    
        # Ensure the buffer is not empty
        if len(buffer) == 0:
            raise ValueError("Buffer length must not be 0")
    
        # Randomly decide count and offset
        count = random.randint(1, num_elements)  # count must be at least 1 and at most num_elements
        offset = random.randint(0, dtype_size * (num_elements - count))
    
        # Invoke torch.frombuffer and return the tensor
        result = torch.frombuffer(buffer, dtype=dtype, count=count, offset=offset)
        return result
    
    
    
    