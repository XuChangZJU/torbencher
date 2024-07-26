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

        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        feature_tensor = torch.randn(input_size)
        bias_int = random.randint(1, 10)

        def fn(o: InputDataClass) -> torch.Tensor:
            res = o.feature + o.bias
            return OutputDataClass(res=res)

        input_data = InputDataClass(feature=feature_tensor, bias=bias_int)
        result = torch.export.export(fn, (input_data,))
        return result
