import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.column_stack)
class TorchColumnstackTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_column_stack_correctness(self):
        # Define the number of tensors to stack
        num_tensors = random.randint(2, 5)
        
        # Generate random dimensions and sizes for tensors
        tensors = []
        for _ in range(num_tensors):
            # Randomly decide the dimension of each tensor (0D, 1D, or 2D)
            dim = random.randint(0, 2)
            if dim == 0:
                # Scalar value converted to a tensor
                tensor = torch.tensor(random.uniform(0.1, 10.0))
            elif dim == 1:
                # 1D tensor with random size
                size = random.randint(1, 5)
                tensor = torch.randn(size)
            else:
                # 2D tensor with random size
                size1 = random.randint(1, 5)
                size2 = random.randint(1, 5)
                tensor = torch.randn(size1, size2)
            tensors.append(tensor)
            
        # Perform the column_stack operation
        result = torch.column_stack(tensors)
        return result
    
    
    
    