import torch
import random
from torch.fx import symbolic_trace, subgraph_rewriter

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.fx.replace_pattern)
class TorchFxReplacepatternTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_replace_pattern_correctness(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
    
            def forward(self, x, w1, w2):
                m1 = torch.cat([w1, w2]).sum()
                m2 = torch.cat([w1, w2]).sum()
                return x + torch.max(m1) + torch.max(m2)
        
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1,5)
        input_size=[num_of_elements_each_dim for i in range(dim)] 
    
        x = torch.randn(input_size)
        w1 = torch.randn(input_size)
        w2 = torch.randn(input_size)
    
        def pattern(w1, w2):
            return torch.cat([w1, w2]).sum()
    
        def replacement(w1, w2):
            return torch.stack([w1, w2])
    
        traced_module = symbolic_trace(M())
        result = subgraph_rewriter.replace_pattern(traced_module, pattern, replacement)
        return result
    