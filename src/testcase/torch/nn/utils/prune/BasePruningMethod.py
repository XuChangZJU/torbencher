import torch
import random
import torch.nn.utils.prune as prune


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.utils.prune.BasePruningMethod)
class TorchNnUtilsPruneBasepruningmethodTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
        def compute_mask(self, t, default_mask):
            mask = default_mask.clone()
            num_elements_to_prune = random.randint(1, t.numel())
            indices_to_prune = torch.randperm(t.numel())[:num_elements_to_prune]
            mask.view(-1)[indices_to_prune] = 0
            return mask
    