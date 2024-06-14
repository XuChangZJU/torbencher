import torch
import random
from torch.nn import Linear
from torch.nn.utils import prune


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.utils.prune.remove)
class TorchNnUtilsPruneRemoveTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_remove_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1,5) # Random number of elements each dimension
        input_size=[num_of_elements_each_dim for i in range(dim)] 
    
        # Create a Linear layer for testing
        linear_layer = Linear(in_features=random.randint(1, 10), out_features=random.randint(1, 10))
    
        # Apply random unstructured pruning to the weight parameter
        prune.random_unstructured(linear_layer, name="weight", amount=0.5) # amount is between 0 and 1
    
        # Remove the pruning reparameterization
        module_after_remove = prune.remove(linear_layer, name="weight")
    
        return module_after_remove
    