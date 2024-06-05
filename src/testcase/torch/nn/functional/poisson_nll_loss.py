
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.poisson_nll_loss)
class PoissonNLLLossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_poisson_nll_loss_correctness(self):
        input_data = torch.randn(10, 10)
        target = torch.randint(0, 10, (10, 10))
        log_input = random.choice([True, False])
        full = random.choice([True, False])
        reduction = random.choice(['mean', 'sum', 'none'])
        result = torch.nn.functional.poisson_nll_loss(input_data, target, log_input, full, reduction)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_poisson_nll_loss_large_scale(self):
        input_data = torch.randn(1000, 1000)
        target = torch.randint(0, 1000, (1000, 1000))
        log_input = random.choice([True, False])
        full = random.choice([True, False])
        reduction = random.choice(['mean', 'sum', 'none'])
        result = torch.nn.functional.poisson_nll_loss(input_data, target, log_input, full, reduction)
        return result

