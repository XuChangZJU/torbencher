import torch
import random
from torch.utils.cpp_extension import is_ninja_available

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.utils.cpp_extension.is_ninja_available)
class TorchUtilsCppUextensionIsUninjaUavailableTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_is_ninja_available(self):
        # Check if ninja is available
        ninja_available = is_ninja_available()
        return ninja_available
