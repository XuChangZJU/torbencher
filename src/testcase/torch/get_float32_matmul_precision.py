import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.get_float32_matmul_precision)
class TorchGetfloat32matmulprecisionTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_get_float32_matmul_precision_correctness(self):
        # Retrieve current float32 matrix multiplication precision
        precision = torch.get_float32_matmul_precision()
        return precision

    def test_set_and_get_float32_matmul_precision(self):
        # Testing all possible values for float32 matrix multiplication precision

        precisions = ["highest", "high", "medium"]
        # Loop through all possible values and set them
        for expected_precision in precisions:
            torch.set_float32_matmul_precision(expected_precision)
            current_precision = torch.get_float32_matmul_precision()

            # Check if the set value matches the retrieved value
            assert current_precision == expected_precision, f"Expected {expected_precision}, but got {current_precision}"

            # Generating two random tensors for matrix multiplication
            # dim1 = random.randint(2, 4)
            # dim2 = random.randint(2, 4)
            # tensor_size1 = [random.randint(2, 5) for _ in range(dim1)]
            # tensor_size2 = tensor_size1[:-1] + [random.randint(2, 5)]
            # Generating two random tensors for matrix multiplication
            batch_dims = [random.randint(2, 5) for _ in range(random.randint(1, 3))]
            rows = random.randint(2, 5)
            common_dim = random.randint(2, 5)
            cols = random.randint(2, 5)

            tensor_size1 = batch_dims + [rows, common_dim]
            tensor_size2 = batch_dims + [common_dim, cols]

            matrix1 = torch.randn(tensor_size1)
            matrix2 = torch.randn(tensor_size2)

            # Perform matrix multiplication to see if the precision setting shows any visible difference
            result = torch.matmul(matrix1, matrix2)

        return result
