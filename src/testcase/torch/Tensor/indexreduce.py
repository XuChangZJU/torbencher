import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.indexreduce)
class TorchTensorIndexreduceTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_index_reduce_correctness(self):
    # Randomly choose dimension for the operation
    dim = random.randint(0, 2)
    
    # Randomly choose the size of the tensor
    num_of_elements_each_dim = random.randint(2, 5)
    tensor_size = [num_of_elements_each_dim for _ in range(3)]
    
    # Create the self tensor
    self_tensor = torch.randn(tensor_size)
    
    # Create the source tensor with the same size as self tensor
    source_tensor = torch.randn(tensor_size)
    
    # Create the index tensor with the same size as the dimension of source tensor
    index_size = tensor_size[dim]
    index_tensor = torch.randint(0, tensor_size[dim], (index_size,), dtype=torch.int64)
    
    # Randomly choose a reduction operation
    reduce_ops = ['prod', 'mean', 'amax', 'amin']
    reduce_op = random.choice(reduce_ops)
    
    # Perform the index_reduce_ operation
    result = self_tensor.index_reduce_(dim, index_tensor, source_tensor, reduce_op)
    
    return result
