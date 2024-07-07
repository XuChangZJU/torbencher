import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.autograd.gradcheck.GradcheckError)
class TorchAutogradGradcheckGradcheckerrorTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_gradcheckerror_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]
    
        # Generate random tensor
        tensor = torch.randn(input_size, requires_grad=True)
        
        # Define a simple function to test gradcheck
        def simple_function(x):
            return x ** 2
    
        # Perform gradcheck
        try:
            torch.autograd.gradcheck(simple_function, (tensor,))
            result = "Gradcheck passed"
        except RuntimeError as e:
            result = f"Gradcheck failed: {str(e)}"
        
        return result
    
    
    
    