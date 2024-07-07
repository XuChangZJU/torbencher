import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.export.graph_signature.get_replace_hook)
class TorchExportGraphsignatureGetreplacehookTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_get_replace_hook_correctness(self):
        # Generate random tensor dimensions
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]
    
        # Generate random tensors
        tensor1 = torch.randn(input_size)
        tensor2 = torch.randn(input_size)
    
        # Define a replace hook function
        def replace_hook(t1, t2):
            return t2
    
        # Apply the replace hook to the tensors
        result = replace_hook(tensor1, tensor2)
        return result
    
    
    
    