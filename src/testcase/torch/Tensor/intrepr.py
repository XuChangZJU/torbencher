import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.intrepr)
class TorchTensorIntreprTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_int_repr_correctness(self):
    # Randomly generate the dimension of the tensor
    dim = random.randint(1, 4)
    # Randomly generate the number of elements in each dimension
    num_of_elements_each_dim = random.randint(1, 5)
    # Create a list of input sizes for the tensor
    input_size = [num_of_elements_each_dim for i in range(dim)]
    
    # Create a random quantized tensor
    quantized_tensor = torch.quantize_per_tensor(torch.randn(input_size), scale=random.uniform(0.1, 1.0), zero_point=random.randint(0, 255), dtype=torch.quint8)
    
    # Get the underlying uint8_t values of the quantized tensor
    result = quantized_tensor.int_repr()
    
    return result
