import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.cosinesimilarity)
class TorchNnFunctionalCosinesimilarityTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cosine_similarity_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim + 1)]  # Ensure tensors have at least one dimension more than `dim`
    
        tensor1 = torch.randn(input_size)
        tensor2 = torch.randn(input_size)
        result = torch.nn.functional.cosine_similarity(tensor1, tensor2, dim=dim)
        return result
    