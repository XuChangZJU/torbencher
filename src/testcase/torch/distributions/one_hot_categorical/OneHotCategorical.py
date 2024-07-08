import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.distributions.one_hot_categorical.OneHotCategorical)
class TorchDistributionsOnehotcategoricalOnehotcategoricalTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_one_hot_categorical_probs_correctness(self):
        # Define the parameters for the OneHotCategorical distribution
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        probs = torch.rand(input_size)  # Generate random probabilities
        probs = probs / probs.sum(-1, keepdim=True)  # Normalize probabilities to sum to 1
    
        # Create a OneHotCategorical distribution
        m = torch.distributions.one_hot_categorical.OneHotCategorical(probs)
    
        # Sample from the distribution
        sample = m.sample()
        
        # Return the sample
        return sample
    def test_one_hot_categorical_logits_correctness(self):
        # Define the parameters for the OneHotCategorical distribution
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        logits = torch.randn(input_size)  # Generate random logits
    
        # Create a OneHotCategorical distribution
        m = torch.distributions.one_hot_categorical.OneHotCategorical(logits=logits)
    
        # Sample from the distribution
        sample = m.sample()
    
        # Return the sample
        return sample
    