
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.profiler.register_optimizer_step_post_hook)
class TorchRegisterOptimizerStepPostHookTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.12.0")
    def test_register_optimizer_step_post_hook_correctness(self):
        optimizer = torch.optim.SGD(torch.nn.Linear(10, 10).parameters(), lr=0.01)
        result = torch.profiler.register_optimizer_step_post_hook(optimizer)
        return result

    @test_api_version.larger_than("1.12.0")
    def test_register_optimizer_step_post_hook_large_scale(self):
        optimizer = torch.optim.SGD(torch.nn.Linear(1000, 1000).parameters(), lr=0.01)
        result = torch.profiler.register_optimizer_step_post_hook(optimizer)
        return result

