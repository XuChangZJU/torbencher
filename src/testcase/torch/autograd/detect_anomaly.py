import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.autograd.detect_anomaly)
class TorchAutogradDetectanomalyTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_detect_anomaly_correctness(self):
        class MyFunc(torch.autograd.Function):
            @staticmethod
            def forward(ctx, inp):
                return inp.clone()

            @staticmethod
            def backward(ctx, grad_output):
                # Error during the backward pass
                raise RuntimeError("Some error in backward")
                return grad_output.clone()

        def run_fn(input_tensor):
            output = MyFunc.apply(input_tensor)
            return output.sum()

        # Generate random input tensor
        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]
        input_tensor = torch.rand(input_size, requires_grad=True)

        try:
            with torch.autograd.detect_anomaly():
                output = run_fn(input_tensor)
                output.__init__()
        except RuntimeError as e:
            f"Caught an error during backward pass: {e}"
