import torch
import random
from torch.utils.tensorboard import SummaryWriter

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.utils.tensorboard.writer.add_graph)
class TorchUtilsTensorboardWriterAddgraphTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_add_graph_correctness(self):
        # Create a random neural network model
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                input_size = random.randint(1, 10)
                hidden_size = random.randint(1, 10)
                output_size = random.randint(1, 10)
                self.fc1 = torch.nn.Linear(input_size, hidden_size)
                self.fc2 = torch.nn.Linear(hidden_size, output_size)
    
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                return x
    
        model = SimpleModel()
        
        # Create a random input tensor
        input_size = model.fc1.in_features
        input_tensor = torch.randn(1, input_size)
        
        # Initialize SummaryWriter
        writer = SummaryWriter()
        
        # Add graph to tensorboard
        writer.add_graph(model, input_tensor)
        
        # Close the writer
        writer.close()
    