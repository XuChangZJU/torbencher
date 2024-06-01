import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.cosine_embedding_loss)
class TorchNNFunctionalCosineEmbeddingLossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cosine_embedding_loss_4d(self, input=None):
        if input is not None:
            result = torch.nn.functional.cosine_embedding_loss(input[0], input[1], input[2], input[3], input[4])
            return [result, input]
        input1 = torch.randn(100, 128)
        input2 = torch.randn(100, 128)
        target = torch.randn(100)
        margin = 0.0
        reduction = 'mean'
        result = torch.nn.functional.cosine_embedding_loss(input1, input2, target, margin, reduction)
        return [result, [input1, input2, target, margin, reduction]]

