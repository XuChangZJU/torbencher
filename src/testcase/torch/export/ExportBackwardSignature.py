import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.export.ExportBackwardSignature)
class TorchExportExportbackwardsignatureTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_ExportBackwardSignature_correctness(self):
        # Generate random parameters for ExportBackwardSignature
        num_gradients_to_parameters = random.randint(1, 5)
        gradients_to_parameters = {f"grad{i}": f"param{i}" for i in range(num_gradients_to_parameters)}  # type: Dict[str, str]
        
        num_gradients_to_user_inputs = random.randint(1, 5)
        gradients_to_user_inputs = {f"grad{i}": f"input{i}" for i in range(num_gradients_to_user_inputs)}  # type: Dict[str, str]
        
        loss_output = "loss"  # type: str
    
        # Create an instance of ExportBackwardSignature
        export_backward_signature = torch.export.ExportBackwardSignature(gradients_to_parameters, gradients_to_user_inputs, loss_output)
    
        return export_backward_signature
    
    
    
    