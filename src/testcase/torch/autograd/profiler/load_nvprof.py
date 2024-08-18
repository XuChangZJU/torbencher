import os
import random
import unittest

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.autograd.profiler.load_nvprof)
class TorchAutogradProfilerLoadUnvprofTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    @unittest.skip("需要实际文件&官网没有找到参考")
    def test_load_nvprof_correctness(self):
        # Generate a random file path for the nvprof trace
        file_path = f"/tmp/nvprof_trace_{random.randint(1, 1000)}.nvvp"

        # Since we cannot generate a real nvprof trace file in this test, we assume the file exists
        # and is valid for the purpose of this test case.
        # In a real-world scenario, you would need to generate or use an existing nvprof trace file.

        # Check if the file exists to avoid errors
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        result = torch.autograd.profiler.load_nvprof(file_path)
        return result
