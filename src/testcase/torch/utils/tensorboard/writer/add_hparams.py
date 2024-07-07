import torch
import random
from torch.utils.tensorboard import SummaryWriter


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.utils.tensorboard.writer.add_hparams)
class TorchUtilsTensorboardWriterAddhparamsTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_add_hparams_correctness(self):
        # Randomly generate hyperparameters
        hparam_keys = ['lr', 'batch_size', 'momentum']
        hparam_values = [random.uniform(0.001, 0.1), random.randint(16, 128), random.uniform(0.5, 0.99)]
        hparams = dict(zip(hparam_keys, hparam_values))
        
        # Randomly generate metrics
        metric_keys = ['accuracy', 'loss']
        metric_values = [random.uniform(0.5, 1.0), random.uniform(0.0, 1.0)]
        metrics = dict(zip(metric_keys, metric_values))
        
        # Create a SummaryWriter instance
        writer = SummaryWriter()
        
        # Add hyperparameters and metrics to the writer
        writer.add_hparams(hparams, metrics)
        
        # Close the writer
        writer.close()
    
    
    
    