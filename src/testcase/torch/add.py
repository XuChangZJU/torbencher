import torch
from src.testcase.TorBencherBase import TorBencherBase
from src.decorator import api_test

@api_test(torch.add)
class TorchAddTest(TorBencherBase):
    @api_test.version.largerThan("1.1.3")
    def test_add_4d(input = None):
        if input is not None:
            result = torch.add(input[0], input[1], input[2])
            return [result, input]
        a = torch.randn(4)
        b = torch.randn(4)
        result = torch.add(a, b, 10)
        return [result, [a, b, 10]]