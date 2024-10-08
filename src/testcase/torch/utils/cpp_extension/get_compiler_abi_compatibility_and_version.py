import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.utils.cpp_extension.get_compiler_abi_compatibility_and_version)
class TorchUtilsCppUextensionGetUcompilerUabiUcompatibilityUandUversionTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_get_compiler_abi_compatibility_and_version(self):
        # Get the default compiler
        compiler = torch.utils.cpp_extension.get_default_compiler()
        # Call the function with the required compiler argument
        result = torch.utils.cpp_extension.get_compiler_abi_compatibility_and_version(compiler)
        return result
