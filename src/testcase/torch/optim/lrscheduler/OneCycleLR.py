import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.optim.lrscheduler.OneCycleLR)
class TorchOptimLrschedulerOnecyclelrTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_OneCycleLR_correctness(self):
        # Define the parameters for the optimizer and scheduler
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        model_params = torch.randn(input_size, requires_grad=True)
        learning_rate = random.uniform(0.001, 0.1)  # Random learning rate between 0.001 and 0.1
        optimizer = torch.optim.SGD([model_params], lr=learning_rate)
        max_lr = random.uniform(learning_rate, 0.1)  # max_lr should be greater than initial lr
        steps_per_epoch = random.randint(10, 100)  # Random steps per epoch between 10 and 100
        epochs = random.randint(1, 10)  # Random number of epochs between 1 and 10
    
        # Create the OneCycleLR scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, steps_per_epoch=steps_per_epoch, epochs=epochs)
    
        # Loop through the epochs and batches, updating the learning rate and recording it
        learning_rates = []
        for epoch in range(epochs):
            for batch in range(steps_per_epoch):
                optimizer.step()
                scheduler.step()
                learning_rates.append(optimizer.param_groups[0]['lr'])
    
        return learning_rates
    