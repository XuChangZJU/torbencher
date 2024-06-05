
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.one_hot)
class OneHotTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_one_hot_correctness(self):
        input_data = torch.randint(0, 10, (10,))
        num_classes = random.randint(10, 20)
        result = torch.nn.functional.one_hot(input_data, num_classes)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_one_hot_large_scale(self):
        input_data = torch.randint(0, 1000, (1000,))
        num_classes = random.randint(1000, 2000)
        result = torch.nn.functional.one_hot(input_data, num_classes)
        return result

