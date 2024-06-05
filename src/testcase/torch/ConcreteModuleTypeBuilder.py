
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.ConcreteModuleTypeBuilder)
class TorchConcreteModuleTypeBuilderTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_concretemoduletypebuilder_correctness(self):
        result = torch.ConcreteModuleTypeBuilder()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_concretemoduletypebuilder_large_scale(self):
        result = torch.ConcreteModuleTypeBuilder()
        return result

