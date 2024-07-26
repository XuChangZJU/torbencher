import torch
import random
from torch.utils.tensorboard import SummaryWriter

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.utils.tensorboard.writer.add_scalar)
class TorchUtilsTensorboardWriterAddscalarTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_add_scalar_correctness(self):
        writer = SummaryWriter()
        
        # Randomly generate a tag name
        tag = f"test_scalar_{random.randint(1, 100)}"
        
        # Randomly generate a scalar value
        scalar_value = random.uniform(0.1, 100.0)
        
        # Randomly generate a global step
        global_step = random.randint(1, 100)
        
        # Add scalar to the writer
        writer.add_scalar(tag, scalar_value, global_step)
        
        # Close the writer
        writer.close()
        
        return f"Scalar with tag {tag}, value {scalar_value}, and step {global_step} added."
    