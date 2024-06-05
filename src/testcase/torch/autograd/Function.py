
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.autograd.Function)
class TorchAutogradFunctionTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_function_correctness(self):
        class MyFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                ctx.save_for_backward(input)
                return input * 2

            @staticmethod
            def backward(ctx, grad_output):
                input, = ctx.saved_tensors
                return grad_output * 2

        input = torch.randn(random.randint(1, 10), requires_grad=True)
        result = MyFunction.apply(input)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_function_large_scale(self):
        class MyFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                ctx.save_for_backward(input)
                return input * 2

            @staticmethod
            def backward(ctx, grad_output):
                input, = ctx.saved_tensors
                return grad_output * 2

        input = torch.randn(random.randint(1000, 10000), requires_grad=True)
        result = MyFunction.apply(input)
        return result


