import torch
import random
import torch.nn.utils.prune as prune


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.utils.prune.BasePruningMethod)
class TorchNnUtilsPruneBasepruningmethodTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
        def compute_mask(self, t, default_mask):
            mask = default_mask.clone()
            num_elements_to_prune = random.randint(1, t.numel())
            indices_to_prune = torch.randperm(t.numel())[:num_elements_to_prune]
            mask.view(-1)[indices_to_prune] = 0
            return mask
    
        def apply(self, module, name):
            mask = self.compute_mask(getattr(module, name), torch.ones_like(getattr(module, name)))
            module.register_buffer(name + '_mask', mask)
            module._parameters[name] = getattr(module, name) * mask
    
    def test_custom_pruning_method_correctness(self):
        # Randomly generate tensor dimensions
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]
    
        # Create a random tensor
        tensor = torch.randn(input_size)
    
        # Create a dummy module with a parameter to prune
        module = torch.nn.Linear(in_features=tensor.numel(), out_features=1)
        module.weight.data = tensor.view(1, -1)
    
        # Apply custom pruning method
        pruning_method = CustomPruningMethod()
        prune.custom_from_mask(module, name='weight', mask=pruning_method.compute_mask(module.weight, torch.ones_like(module.weight)))
    
        # Check the pruned tensor
        pruned_tensor = module.weight * module.weight_mask
        return pruned_tensor
    
    
    
    