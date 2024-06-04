
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.instance_norm)
class TorchNNFunctionalInstanceNormTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_instance_norm(self):
        
        a = torch.randn(20, 100, 35, 45)
        b = torch.zeros(100)
        c = torch.ones(100)
        d = None
        e = None
        result = torch.nn.functional.instance_norm(
            a,
            running_mean=b,
            running_var=c,
            weight=d,
            bias=e,
            use_input_stats=True,
            momentum=0.1,
            eps=1e-05,
        )
        return result


