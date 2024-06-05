
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.autograd.NestedIOFunction)
class TorchAutogradNestedIOFunctionTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_nestediofunction_correctness(self):
        class MyNestedIOFunction(torch.autograd.NestedIOFunction):
            @staticmethod
            def forward(ctx, input1, input2):
                ctx.save_for_backward(input1, input2)
                return input1 + input2

            @staticmethod
            def backward(ctx, grad_output):
                input1, input2 = ctx.saved_tensors
                return grad_output, grad_output

        input1 = torch.randn(random.randint(1, 10), requires_grad=True)
        input2 = torch.randn(random.randint(1, 10), requires_grad=True)
        result = MyNestedIOFunction.apply(input1, input2)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_nestediofunction_large_scale(self):
        class MyNestedIOFunction(torch.autograd.NestedIOFunction):
            @staticmethod
            def forward(ctx, input1, input2):
                ctx.save_for_backward(input1, input2)
                return input1 + input2

            @staticmethod
            def backward(ctx, grad_output):
                input1, input2 = ctx.saved_tensors
                return grad_output, grad_output

        input1 = torch.randn(random.randint(1000, 10000), requires_grad=True)
        input2 = torch.randn(random.randint(1000, 10000), requires_grad=True)
        result = MyNestedIOFunction.apply(input1, input2)
        return result


