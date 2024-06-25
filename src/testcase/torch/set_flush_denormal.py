import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.set_flush_denormal)
class TorchSetflushdenormalTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_set_flush_denormal_correctness(self):
        # Randomly enable or disable flush denormal mode
        mode = random.choice([True, False])
        result = torch.set_flush_denormal(mode)
    
        # Verify if the function returns True when supported and the setting is applied
        if result:
            # Create a very small denormal number tensor
            denormal_tensor = torch.tensor([1e-323], dtype=torch.float64)
            if mode:
                # When flushing denormals, tensor should be zero
                assert torch.all(denormal_tensor == 0), "Expected denormal tensor to be flushed to zero"
            else:
                # When not flushing denormals, tensor should retain its denormal value
                assert torch.all(denormal_tensor != 0), "Expected denormal tensor to retain its value"
    
        return result
    
    
    
    