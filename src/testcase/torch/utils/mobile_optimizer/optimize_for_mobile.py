import torch
import random
from torch.utils.mobile_optimizer import optimize_for_mobile

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.utils.mobile_optimizer.optimize_for_mobile)
class TorchUtilsMobileoptimizerOptimizeformobileTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_optimize_for_mobile_correctness(self):
        # Generate a random model
        input_size = [random.randint(1, 5) for _ in range(random.randint(1, 4))]
        model = torch.nn.Sequential(
            torch.nn.Linear(input_size[-1], random.randint(1, 10)),
            torch.nn.ReLU(),
            torch.nn.Linear(random.randint(1, 10), random.randint(1, 10))
        )
        
        # Generate a random input tensor
        input_tensor = torch.randn(input_size)
        
        # Optimize the model for mobile
        scripted_model = torch.jit.script(model)
        optimized_model = optimize_for_mobile(scripted_model)
        
        # Run the model and the optimized model with the same input
        original_output = model(input_tensor)
        optimized_output = optimized_model(input_tensor)
        
        return original_output, optimized_output
    