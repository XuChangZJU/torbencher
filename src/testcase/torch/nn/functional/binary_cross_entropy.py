
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.binary_cross_entropy)
class TorchNNFunctionalBinaryCrossEntropyTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_binary_cross_entropy_common(self):
        
        a = torch.randn(3, 2)
        b = torch.randn(3, 2)
        c = None
        d = True
        e = True
        f = 'mean'
        result = torch.nn.functional.binary_cross_entropy(a, b, weight=c, size_average=d, reduce=e, reduction=f)
        return result


