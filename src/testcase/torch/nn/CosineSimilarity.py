import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.CosineSimilarity)
class TorchNnCosinesimilarityTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cosine_similarity_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim + 1)]  # Ensure tensors have at least one dimension for cosine similarity
    
        tensor1 = torch.randn(input_size)
        tensor2 = torch.randn(input_size)
        eps = random.uniform(1e-10, 1e-6)  # Random epsilon value between 1e-10 and 1e-6
    
        cosine_similarity = torch.nn.CosineSimilarity(dim=dim, eps=eps)
        result = cosine_similarity(tensor1, tensor2)
        return result
    