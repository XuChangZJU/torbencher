import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.fmin)
class TorchFminTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_fmin_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]
    
        tensor1 = torch.randn(input_size)
        tensor2 = torch.randn(input_size)
        
        # Randomly introducing NaN values in the tensor1 and tensor2
        nan_indices_tensor1 = [random.uniform(0, 1) < 0.3 for _ in range(torch.numel(tensor1))]
        nan_indices_tensor2 = [random.uniform(0, 1) < 0.3 for _ in range(torch.numel(tensor2))]
        
        for idx, is_nan in enumerate(nan_indices_tensor1):
            if is_nan:
                tensor1.view(-1)[idx] = float('nan')
    
        for idx, is_nan in enumerate(nan_indices_tensor2):
            if is_nan:
                tensor2.view(-1)[idx] = float('nan')
        
        result = torch.fmin(tensor1, tensor2)
        return result
    
    
    
    