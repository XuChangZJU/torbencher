import torch


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.autograd.Function)
class TorchAutogradFunctionTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_custom_function_correctness(self):
        """Tests correctness of a custom autograd.Function with random inputs."""
        dim = random.randint(1, 4)  # Random dimension for tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Random tensor with requires_grad=True for autograd
        tensor1 = torch.randn(input_size, requires_grad=True)
    
        # Define a simple custom function that squares the input
        class Square(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                result = input * input
                ctx.save_for_backward(input)
                return result
    
            @staticmethod
            def backward(ctx, grad_output):
                input, = ctx.saved_tensors
                return grad_output * 2 * input
    
        # Apply the custom function
        result = Square.apply(tensor1)
    
        # Check correctness with autograd
        expected_result = tensor1 * tensor1
        assert torch.allclose(result, expected_result)
        
    
    
    