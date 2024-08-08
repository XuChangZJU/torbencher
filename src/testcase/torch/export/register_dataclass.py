from dataclasses import dataclass

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.export.register_dataclass)
class TorchExportRegisterUdataclassTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_register_dataclass_correctness(self):
        @dataclass
        class InputDataClass:
            feature: torch.Tensor
            bias: int

        @dataclass
        class OutputDataClass:
            res: torch.Tensor

        torch.export.register_dataclass(InputDataClass)
        torch.export.register_dataclass(OutputDataClass)
