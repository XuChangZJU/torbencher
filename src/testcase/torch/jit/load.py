import torch
import random
import io

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.jit.load)
class TorchJitLoadTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_load_correctness(self):
        """
        This test checks the correctness of torch.jit.load by saving a simple model,
        loading it back, and verifying the loaded model's output matches the original.
        """
        class SimpleModel(torch.nn.Module):
            def __init__(self, in_features, out_features):
                super(SimpleModel, self).
                self.linear = torch.nn.Linear(in_features, out_features)
    
            def forward(self, x):
                return self.linear(x)
    
        # Generate random input data
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        input_tensor = torch.randn(input_size)
    
        # Create a simple model
        in_features = input_tensor.shape[-1]
        out_features = random.randint(1, 10)
        model = SimpleModel(in_features, out_features)
        scripted_model = torch.jit.script(model)
    
        # Save the model
        buffer = io.BytesIO()
        torch.jit.save(scripted_model, buffer)
    
        # Load the model
        buffer.seek(0)
        loaded_model = torch.jit.load(buffer)
    
        # Verify the loaded model's output matches the original
        original_output = model(input_tensor)
        loaded_output = loaded_model(input_tensor)
        return original_output, loaded_output
    