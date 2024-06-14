import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.AdaptiveLogSoftmaxWithLoss)
class TorchNnAdaptivelogsoftmaxwithlossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_adaptive_log_softmax_with_loss_correctness(self):
        # Randomly generate the number of features and classes
        in_features = random.randint(10, 100)
        n_classes = random.randint(10, 100)
        
        # Randomly generate cutoffs ensuring they are in increasing order and within the range of n_classes
        cutoffs = sorted(random.sample(range(1, n_classes), random.randint(1, n_classes // 2)))
        
        # Create the AdaptiveLogSoftmaxWithLoss module
        adaptive_log_softmax = torch.nn.AdaptiveLogSoftmaxWithLoss(in_features, n_classes, cutoffs)
        
        # Randomly generate input tensor of shape (N, in_features)
        N = random.randint(1, 10)
        input_tensor = torch.randn(N, in_features)
        
        # Randomly generate target tensor of shape (N) with values in the range [0, n_classes)
        target_tensor = torch.randint(0, n_classes, (N,))
        
        # Compute the output and loss
        output = adaptive_log_softmax(input_tensor, target_tensor)
        
        return output
    
    
    
    