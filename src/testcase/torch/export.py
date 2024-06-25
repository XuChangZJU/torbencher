import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.export)
class TorchExportTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_export_correctness(self):
        input_size = [random.randint(1, 3) for _ in range(2)]  # Random dimension for input tensor
        tensor = torch.randn(input_size)
        model = torch.nn.Linear(input_size[1], random.randint(1, 5))  # Random Linear Model with valid dimensions
        
        # Setting up the function to export
        def model_to_export(x):
            return model(x)
        
        # Exporting the model
        scripted_model = torch.jit.script(model_to_export)
        
        return scripted_model
    
    
    
    