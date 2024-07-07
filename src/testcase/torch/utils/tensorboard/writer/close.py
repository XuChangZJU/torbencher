import torch
import random
from torch.utils.tensorboard import SummaryWriter


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.utils.tensorboard.writer.close)
class TorchUtilsTensorboardWriterCloseTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_tensorboard_writer_close(self):
        log_dir = f"runs/test_{random.randint(1, 1000)}"  # Random log directory
        writer = SummaryWriter(log_dir=log_dir)
        
        # Write some random data to the writer
        for i in range(random.randint(1, 10)):  # Random number of data points
            x = torch.randn(random.randint(1, 10))  # Random tensor data
            writer.add_scalar('data/scalar', x.mean().item(), i)
        
        writer.close()
        
        # Check if the writer is closed
        return writer._get_file_writer() is None
    
    
    
    