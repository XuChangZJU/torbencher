import torch
from torch.utils.cpp_extension import is_ninja_available

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.utils.cpp_extension.verify_ninja_availability)
class TorchUtilsCppUextensionVerifyUninjaUavailabilityTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_verify_ninja_availability(self):
        # This function does not take any parameters and returns a boolean value
        result = is_ninja_available()
        return result
