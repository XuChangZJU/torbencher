import random

import torch
from torch.utils.cpp_extension import CppExtension

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.utils.cpp_extension.CppExtension)
class TorchUtilsCppUextensionCppextensionTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_cpp_extension_correctness(self):
        # Randomly generate the name of the extension
        extension_name = f"extension_{random.randint(1, 1000)}"

        # Randomly generate the source files list
        num_files = random.randint(1, 3)  # Random number of source files between 1 and 3
        source_files = [f"source_file_{i}.cpp" for i in range(num_files)]

        # Create the CppExtension object
        cpp_extension = CppExtension(extension_name, source_files)

        # Return the created CppExtension object to verify correctness
        return cpp_extension
