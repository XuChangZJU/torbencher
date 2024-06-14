import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.ReplicationPad2d)
class TorchNnReplicationpad2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_ReplicationPad2d_correctness(self):
        # Randomly generate input tensor dimension
        dim = random.randint(3, 4)  
        # Randomly generate the number of elements in each dimension
        num_of_elements_each_dim = random.randint(1, 5) 
        # Generate input_size list for torch.randn
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate random input tensor
        input_tensor = torch.randn(input_size)
        # Generate random padding size
        padding_size = random.randint(1, 3)
        # Generate ReplicationPad2d module
        replicationpad2d = torch.nn.ReplicationPad2d(padding_size)
        # Calculate result
        result = replicationpad2d(input_tensor)
        # Return result
        return result
    
    
    
    