import torch
import random
from torch.cuda.amp import GradScaler

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.cuda.amp.GradScaler)
class TorchCudaAmpGradscalerTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_grad_scaler_correctness(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please ensure that your PyTorch installation is compiled with CUDA support.")
    
        # Randomly generate tensor dimensions and size
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]
    
        # Randomly generate model parameters
        model = torch.nn.Linear(num_of_elements_each_dim, num_of_elements_each_dim).cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss_fn = torch.nn.MSELoss()
    
        # Randomly generate input and target tensors
        input_tensor = torch.randn(input_size).cuda()
        target_tensor = torch.randn(input_size).cuda()
    
        # Create a GradScaler instance
        scaler = GradScaler()
    
        # Forward pass
        optimizer.zero_grad()
        output = model(input_tensor)
        loss = loss_fn(output, target_tensor)
    
        # Scale the loss and perform backward pass
        scaled_loss = scaler.scale(loss)
        scaled_loss.backward()
    
        # Perform optimizer step
        scaler.step(optimizer)
    
        # Update the scaler
        scaler.update()
    
        return scaled_loss
    