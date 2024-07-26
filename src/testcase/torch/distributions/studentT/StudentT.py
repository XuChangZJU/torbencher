import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.distributions.studentT.StudentT)
class TorchDistributionsStudenttStudenttTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_studentT_correctness(self):
        # Randomly generate degrees of freedom, mean, and scale
        df = torch.tensor([random.uniform(0.1, 10.0)])  # Degrees of freedom between 0.1 and 10.0
        loc = torch.tensor([random.uniform(-5.0, 5.0)])  # Mean between -5.0 and 5.0
        scale = torch.tensor([random.uniform(0.1, 5.0)])  # Scale between 0.1 and 5.0
    
        # Create Student's t-distribution with the generated parameters
        student_t_distribution = torch.distributions.studentT.StudentT(df, loc, scale)
        
        # Sample from the distribution
        sample = student_t_distribution.sample()
        return sample
    