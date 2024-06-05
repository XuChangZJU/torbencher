
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.AdaptiveLogSoftmaxWithLoss)
class TorchAdaptiveLogSoftmaxWithLossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_adaptivelogsoftmaxwithloss_correctness(self):
        n_classes = random.randint(10, 100)
        in_features = random.randint(1, n_classes)
        cut_off = random.randint(1, n_classes)
        input_tensor = torch.randn(1, in_features)
        target = torch.randint(0, n_classes, (1,))
        adaptive_log_softmax = torch.nn.AdaptiveLogSoftmaxWithLoss(in_features, n_classes, cut_off=cut_off)
        result = adaptive_log_softmax(input_tensor, target)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_adaptivelogsoftmaxwithloss_large_scale(self):
        n_classes = random.randint(1000, 10000)
        in_features = random.randint(100, n_classes)
        cut_off = random.randint(100, n_classes)
        input_tensor = torch.randn(100, in_features)
        target = torch.randint(0, n_classes, (100,))
        adaptive_log_softmax = torch.nn.AdaptiveLogSoftmaxWithLoss(in_features, n_classes, cut_off=cut_off)
        result = adaptive_log_softmax(input_tensor, target)
        return result

