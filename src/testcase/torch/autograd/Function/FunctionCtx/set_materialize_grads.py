import torch
import random
from torch.autograd import Function
from torch.autograd.function import once_differentiable


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.autograd.function.FunctionCtx.set_materialize_grads)
class TorchAutogradFunctionFunctionctxSetmaterializegradsTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
        def forward(ctx, x):
            ctx.set_materialize_grads(False)
            ctx.save_for_backward(x)
            return x.clone(), x.clone()
    
        @staticmethod
        @once_differentiable
        def backward(ctx, g1, g2):
            x, = ctx.saved_tensors
            grad_input = torch.zeros_like(x)
            if g1 is not None:
                grad_input += g1
            if g2 is not None:
                grad_input += g2
            return grad_input
    
    class SimpleFunc(Function):
        @staticmethod
        def forward(ctx, x):
            return x.clone(), x.clone()
    
        @staticmethod
        @once_differentiable
        def backward(ctx, g1, g2):
            return g1 + g2
    
    def test_set_materialize_grads_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1,5) # Random number of elements each dimension
        input_size=[num_of_elements_each_dim for i in range(dim)] 
    
        a = torch.randn(input_size, requires_grad=True)
        b, c = MyFunc.apply(a)
        b.sum().
        result1 = a.grad
    
        a = torch.randn(input_size, requires_grad=True)
        b, c = SimpleFunc.apply(a)
        b.sum().
        result2 = a.grad
        return result1, result2
        
    
    
    
    