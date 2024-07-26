import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.cuda.amp.custom_fwd)
class TorchCudaAmpCustomfwdTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_custom_fwd_correctness(self):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. Please ensure that your PyTorch installation is compiled with CUDA support.")

        class MyFunction(torch.autograd.Function):
            @torch.cuda.amp.custom_fwd(cast_inputs=torch.float16)
            def forward(ctx, input_tensor):
                return input_tensor * 2

        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        input_tensor = torch.randn(input_size, device='cuda', dtype=torch.float32)
        result = MyFunction.apply(input_tensor)
        return result
