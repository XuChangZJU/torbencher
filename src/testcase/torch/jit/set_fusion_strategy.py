import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.jit.set_fusion_strategy)
class TorchJitSetfusionstrategyTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_set_fusion_strategy_correctness(self):
        # Randomly generate fusion strategy
        strategy_type = random.choice(["STATIC", "DYNAMIC"])
        strategy_depth = random.randint(1, 5)
        fusion_strategy = [(strategy_type, strategy_depth)]
        
        # Set the fusion strategy
        torch.jit.set_fusion_strategy(fusion_strategy)
    
        # Define a simple model
        class MyModel(torch.nn.Module):
            def __init__(self):
                super(MyModel, self).
                self.linear = torch.nn.Linear(10, 10)
    
            def forward(self, x):
                return self.linear(x)
    
        # Create an instance of the model
        model = MyModel()
    
        # Trace the model
        traced_model = torch.jit.trace(model, torch.randn(1, 10))
    
        # Run an inference
        result = traced_model(torch.randn(1, 10))
        
        return result
    
    
    
    