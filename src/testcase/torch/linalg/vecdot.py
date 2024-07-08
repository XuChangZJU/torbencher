import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.linalg.vecdot)
class TorchLinalgVecdotTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_torch_linalg_vecdot_correctness(self):
        # Define the dimension of the vectors and the batch size
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Generate two random tensors of the same size
        vector1 = torch.randn(input_size)
        vector2 = torch.randn(input_size)
    
        # Compute the dot product using torch.linalg.vecdot
        result = torch.linalg.vecdot(vector1, vector2)
        return result
    