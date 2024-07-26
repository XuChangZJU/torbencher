import torch
import random
from torch.utils.tensorboard import SummaryWriter

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.utils.tensorboard.writer.add_scalars)
class TorchUtilsTensorboardWriterAddscalarsTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_add_scalars_correctness(self):
        writer = SummaryWriter()
    
        # Randomly generate the number of steps
        num_steps = random.randint(1, 10)
    
        # Randomly generate the number of scalars
        num_scalars = random.randint(1, 5)
    
        # Randomly generate scalar values for each step
        scalars_dict = {}
        for i in range(num_scalars):
            scalar_name = f'scalar_{i}'
            scalars_dict[scalar_name] = {f'step_{j}': random.uniform(0.1, 10.0) for j in range(num_steps)}
    
        # Add scalars to the writer for each step
        for step in range(num_steps):
            step_scalars = {scalar_name: scalars_dict[scalar_name][f'step_{step}'] for scalar_name in scalars_dict}
            writer.add_scalars('test_scalars', step_scalars, step)
    
        # Close the writer
        writer.close()
    