import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.export.graph_signature.CustomObjArgument)
class TorchExportGraphsignatureCustomobjargumentTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_custom_obj_argument_correctness(self):
        # Randomly generate a string name for the custom object
        name = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=random.randint(3, 10)))
        
        # Randomly generate a tensor for the custom object
        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]
        tensor_data = torch.randn(input_size)
        
        # Create the custom object argument
        custom_obj_argument = torch.export.graph_signature.CustomObject(name, tensor_data)
        
        # Return the custom object argument
        return custom_obj_argument
    
    
    
    