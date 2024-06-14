import torch
import torch.nn as nn
import torch.nn.functional as F
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.KLDivLoss)
class TorchNnKldivlossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_kl_div_loss_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]
    
        # Generate random input tensor in log-space
        input_tensor = F.log_softmax(torch.randn(input_size), dim=-1)
        # Generate random target tensor
        target_tensor = F.softmax(torch.randn(input_size), dim=-1)
    
        # Initialize KLDivLoss with default reduction 'mean'
        kl_loss = nn.KLDivLoss()
        result_mean = kl_loss(input_tensor, target_tensor)
    
        # Initialize KLDivLoss with reduction 'batchmean'
        kl_loss_batchmean = nn.KLDivLoss(reduction='batchmean')
        result_batchmean = kl_loss_batchmean(input_tensor, target_tensor)
    
        # Initialize KLDivLoss with reduction 'sum'
        kl_loss_sum = nn.KLDivLoss(reduction='sum')
        result_sum = kl_loss_sum(input_tensor, target_tensor)
    
        # Initialize KLDivLoss with reduction 'none'
        kl_loss_none = nn.KLDivLoss(reduction='none')
        result_none = kl_loss_none(input_tensor, target_tensor)
    
        # Initialize KLDivLoss with log_target=True
        log_target_tensor = F.log_softmax(torch.randn(input_size), dim=-1)
        kl_loss_log_target = nn.KLDivLoss(log_target=True)
        result_log_target = kl_loss_log_target(input_tensor, log_target_tensor)
    
        return result_mean, result_batchmean, result_sum, result_none, result_log_target
    