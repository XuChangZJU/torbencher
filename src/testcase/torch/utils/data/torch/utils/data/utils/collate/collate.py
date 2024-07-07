import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.utils.data.torch.utils.data._utils.collate.collate)
class TorchUtilsDataTorchUtilsDataUtilsCollateCollateTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_collate_correctness(self):
        # Generate random parameters for the input list of tensors
        num_tensors = random.randint(1, 10)  # Random number of tensors in the list
        max_tensor_dim = random.randint(1, 4)  # Maximum number of dimensions for each tensor
        
        # Generate a list of random tensors with potentially different sizes
        tensor_list = []
        for _ in range(num_tensors):
            dim = random.randint(1, max_tensor_dim)  # Random dimension for the tensor
            num_of_elements_each_dim = [random.randint(1, 5) for _ in range(dim)]
            tensor_list.append(torch.randn(num_of_elements_each_dim))
        
        result = torch.utils.data._utils.collate.collate(tensor_list)
        return result
    
    
    
    