import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.getnumthreads)
class TorchGetnumthreadsTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_get_num_threads_correctness(self):
    # Torch function to return the number of threads used for parallelizing CPU operations
