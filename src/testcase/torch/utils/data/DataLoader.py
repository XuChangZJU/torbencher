import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.utils.data.DataLoader)
class TorchUtilsDataDataloaderTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_DataLoader_correctness(self):
        # Random parameters for DataLoader
        dataset_size = random.randint(10, 100)  # Random dataset size
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]
        batch_size = random.randint(1, dataset_size)  # Random batch size
    
        # Create random dataset
        dataset = torch.randn(dataset_size, *input_size)
    
        # Create DataLoader
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    
        # Iterate over the DataLoader and return the first batch
        for batch in dataloader:
            return batch
    
    
    
    