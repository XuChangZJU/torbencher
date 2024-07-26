import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.special.airy_ai)
class TorchSpecialAiryaiTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_airy_ai_correctness(self):
        # Randomly generate input tensor data
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        input_tensor = torch.randn(input_size) # input tensor
    
        result = torch.special.airy_ai(input_tensor)
        return result
    