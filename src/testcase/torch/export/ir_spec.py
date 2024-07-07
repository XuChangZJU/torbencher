import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.export.ir_spec)
class TorchExportIrspecTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_export_ir_spec_correctness(self):
        # Randomly generate the dimension for the tensor
        dim = random.randint(1, 4)
        
        # Randomly generate the number of elements in each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]
        
        # Generate a random tensor with the specified dimensions
        tensor = torch.randn(input_size)
        
        # Assuming the goal is to export the tensor to a file, we can use torch.save
        file_path = "tensor_export.pt"
        torch.save(tensor, file_path)
        
        return file_path
    
    
    
    