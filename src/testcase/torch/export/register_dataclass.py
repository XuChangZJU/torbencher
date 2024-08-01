import torch
import random
from dataclasses import dataclass

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.export.register_dataclass)
class TorchExportRegisterdataclassTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
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
