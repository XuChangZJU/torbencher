import random
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.set_default_tensor_type)
class TorchSetdefaulttensortypeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_set_default_tensor_type_correctness(self):
        tensor_types = [torch.FloatTensor, torch.DoubleTensor, torch.HalfTensor]
        chosen_type = random.choice(tensor_types)

        initial_tensor = torch.tensor([1.2, 3.0])  # Should be torch.float32 initially
        initial_dtype = initial_tensor.dtype

        torch.set_default_tensor_type(chosen_type)
        new_tensor = torch.tensor([1.2, 3.0])  # Should be the chosen random type
        new_dtype = new_tensor.dtype

        # Return the initial dtype, new dtype, and a tensor to confirm the change
        return (initial_dtype, new_dtype, new_tensor)
