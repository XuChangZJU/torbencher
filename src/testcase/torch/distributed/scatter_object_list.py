
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.distributed.scatter_object_list)
class TorchScatterObjectListTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_scatter_object_list_correctness(self):
        obj_list = [random.randint(1, 100) for _ in range(random.randint(1, 10))]
        result = torch.distributed.scatter_object_list(obj_list)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_scatter_object_list_large_scale(self):
        obj_list = [random.randint(1, 100) for _ in range(random.randint(100, 1000))]
        result = torch.distributed.scatter_object_list(obj_list)
        return result

