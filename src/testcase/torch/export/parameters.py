import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.export.parameters)
class TorchExportParametersTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_export_parameters_correctness(self):
        # Randomly generate the number of parameters
        num_params = random.randint(1, 5)
        
        # Create a list of random tensors to simulate parameters
        parameters = [torch.randn(random.randint(1, 4), random.randint(1, 4)) for _ in range(num_params)]
        
        # Use torch.save to export the parameters to a file
        torch.save(parameters, 'exported_params.pth')
        
        # Load the parameters back to verify correctness
        exported_params = torch.load('exported_params.pth')
        
        return exported_params
    
    
    
    