import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.backends.opt_einsum.get_opt_einsum)
class TorchBackendsOpteinsumGetopteinsumTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_get_opt_einsum_correctness(self):
        # Randomly choose a valid equation for einsum
        equations = ['ij,jk->ik', 'ij,j->i', 'ijk,kl->ijl', 'ij,jk,kl->il']
        equation = random.choice(equations)
        
        # Determine the shapes of the tensors based on the chosen equation
        if equation == 'ij,jk->ik':
            dim1 = random.randint(1, 4)
            dim2 = random.randint(1, 4)
            dim3 = random.randint(1, 4)
            tensor1 = torch.randn(dim1, dim2)
            tensor2 = torch.randn(dim2, dim3)
        elif equation == 'ij,j->i':
            dim1 = random.randint(1, 4)
            dim2 = random.randint(1, 4)
            tensor1 = torch.randn(dim1, dim2)
            tensor2 = torch.randn(dim2)
        elif equation == 'ijk,kl->ijl':
            dim1 = random.randint(1, 4)
            dim2 = random.randint(1, 4)
            dim3 = random.randint(1, 4)
            dim4 = random.randint(1, 4)
            tensor1 = torch.randn(dim1, dim2, dim3)
            tensor2 = torch.randn(dim3, dim4)
        elif equation == 'ij,jk,kl->il':
            dim1 = random.randint(1, 4)
            dim2 = random.randint(1, 4)
            dim3 = random.randint(1, 4)
            dim4 = random.randint(1, 4)
            tensor1 = torch.randn(dim1, dim2)
            tensor2 = torch.randn(dim2, dim3)
            tensor3 = torch.randn(dim3, dim4)
        
        # Perform the einsum operation using the optimal path
        if equation == 'ij,jk->ik':
            result = torch.einsum(equation, tensor1, tensor2, optimize=True)
        elif equation == 'ij,j->i':
            result = torch.einsum(equation, tensor1, tensor2, optimize=True)
        elif equation == 'ijk,kl->ijl':
            result = torch.einsum(equation, tensor1, tensor2, optimize=True)
        elif equation == 'ij,jk,kl->il':
            result = torch.einsum(equation, tensor1, tensor2, tensor3, optimize=True)
        
        return result
    
    
    
    