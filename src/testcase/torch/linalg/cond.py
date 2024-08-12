import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.linalg.cond)
class TorchLinalgCondTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_torch_linalg_cond_correctness(self):
        # Define the dimension of the matrix
        dim = random.randint(1, 4)
        # Define the size of the matrix
        input_size = [random.randint(1, 5) for _ in range(dim)] * 2
        # Generate a random matrix
        A = torch.randn(input_size)
        # Calculate the condition number of the matrix
        result = torch.linalg.cond(A)
        return result

    def test_torch_linalg_cond_p_fro(self):
        # Define the dimension of the matrix
        dim = random.randint(1, 4)
        # Define the size of the matrix
        input_size = [random.randint(1, 5) for _ in range(dim)] * 2
        # Generate a random matrix
        A = torch.randn(input_size)
        # Calculate the condition number of the matrix using Frobenius norm
        result = torch.linalg.cond(A, 'fro')
        return result

    def test_torch_linalg_cond_p_nuc(self):
        # Define the dimension of the matrix
        dim = random.randint(1, 4)
        # Define the size of the matrix
        input_size = [random.randint(1, 5) for _ in range(dim)] * 2
        # Generate a random matrix
        A = torch.randn(input_size)
        # Calculate the condition number of the matrix using nuclear norm
        result = torch.linalg.cond(A, 'nuc')
        return result

    def test_torch_linalg_cond_p_inf(self):
        # Define the dimension of the matrix
        dim = random.randint(1, 4)
        # Define the size of the matrix
        input_size = [random.randint(1, 5) for _ in range(dim)] * 2
        # Generate a random matrix
        A = torch.randn(input_size)
        # Calculate the condition number of the matrix using inf norm
        result = torch.linalg.cond(A, float('inf'))
        return result

    def test_torch_linalg_cond_p_negative_inf(self):
        # Define the dimension of the matrix
        dim = random.randint(1, 4)
        # Define the size of the matrix
        input_size = [random.randint(1, 5) for _ in range(dim)] * 2
        # Generate a random matrix
        A = torch.randn(input_size)
        # Calculate the condition number of the matrix using -inf norm
        result = torch.linalg.cond(A, float('-inf'))
        return result

    def test_torch_linalg_cond_p_1(self):
        # Define the dimension of the matrix
        dim = random.randint(1, 4)
        # Define the size of the matrix
        input_size = [random.randint(1, 5) for _ in range(dim)] * 2
        # Generate a random matrix
        A = torch.randn(input_size)
        # Calculate the condition number of the matrix using 1-norm
        result = torch.linalg.cond(A, 1)
        return result

    def test_torch_linalg_cond_p_negative_1(self):
        # Define the dimension of the matrix
        dim = random.randint(1, 4)
        # Define the size of the matrix
        input_size = [random.randint(1, 5) for _ in range(dim)] * 2
        # Generate a random matrix
        A = torch.randn(input_size)
        # Calculate the condition number of the matrix using -1-norm
        result = torch.linalg.cond(A, -1)
        return result

    def test_torch_linalg_cond_p_2(self):
        # Define the dimension of the matrix
        dim = random.randint(1, 4)
        # Define the size of the matrix
        input_size = [random.randint(1, 5) for _ in range(dim)] * 2
        # Generate a random matrix
        A = torch.randn(input_size)
        # Calculate the condition number of the matrix using 2-norm
        result = torch.linalg.cond(A, 2)
        return result

    def test_torch_linalg_cond_p_negative_2(self):
        # Define the dimension of the matrix
        dim = random.randint(1, 4)
        # Define the size of the matrix
        input_size = [random.randint(1, 5) for _ in range(dim)] * 2
        # Generate a random matrix
        A = torch.randn(input_size)
        # Calculate the condition number of the matrix using -2-norm
        result = torch.linalg.cond(A, -2)
        return result
