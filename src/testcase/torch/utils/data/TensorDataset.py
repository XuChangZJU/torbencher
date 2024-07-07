import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.utils.data.TensorDataset)
class TorchUtilsDataTensordatasetTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_tensor_dataset_correctness(self):
        # Randomly generate the size of the tensors
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        input_size[0] = random.randint(1, 10) # Make sure the first dimension is the same for all tensors
    
        # Create random tensors with the same size for the first dimension
        tensors = [torch.randn(input_size) for _ in range(random.randint(1, 3))] # Create 1 to 3 tensors
        dataset = torch.utils.data.TensorDataset(*tensors)
        index = random.randint(0, len(dataset) - 1) # Randomly select an index within the dataset
        sample = dataset[index]
        return sample
        
    
    
    
    