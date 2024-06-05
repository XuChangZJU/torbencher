
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.FeatureAlphaDropout)
class TorchFeatureAlphaDropoutTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_featurealphadropout_correctness(self):
        p = random.uniform(0.0, 1.0)
        input_tensor = torch.randn(random.randint(1, 10), random.randint(1, 10))
        feature_alpha_dropout = torch.nn.FeatureAlphaDropout(p=p)
        result = feature_alpha_dropout(input_tensor)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_featurealphadropout_large_scale(self):
        p = random.uniform(0.0, 1.0)
        input_tensor = torch.randn(random.randint(1000, 10000), random.randint(100, 1000))
        feature_alpha_dropout = torch.nn.FeatureAlphaDropout(p=p)
        result = feature_alpha_dropout(input_tensor)
        return result

