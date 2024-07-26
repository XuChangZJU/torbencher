import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.autograd.graph.disable_saved_tensors_hooks)
class TorchAutogradGraphDisablesavedtensorshooksTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_disable_saved_tensors_hooks_correctness(self):
        error_message = "saved tensors default hooks are disabled"  # Error message
        with torch.autograd.graph.disable_saved_tensors_hooks(error_message):
            try:
                with torch.autograd.graph.save_on_cpu():
                    pass
            except RuntimeError as e:
                assert str(e) == error_message
                return e
