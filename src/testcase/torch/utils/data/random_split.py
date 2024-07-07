import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.utils.data.random_split)
class TorchUtilsDataRandomsplitTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_random_split_correctness(self):
        # Randomly generate the size of the dataset
        dataset_size = random.randint(10, 100)
        dataset = list(range(dataset_size))
        
        # Randomly generate the number of splits
        num_splits = random.randint(2, 5)
        
        # Randomly generate the lengths for each split ensuring they sum up to the dataset size
        lengths = [random.randint(1, dataset_size // num_splits) for _ in range(num_splits - 1)]
        lengths.append(dataset_size - sum(lengths))
        
        # Ensure the lengths sum up to the dataset size
        assert sum(lengths) == dataset_size
        
        # Perform the random split
        splits = torch.utils.data.random_split(dataset, lengths)
        
        # Return the lengths of each split to verify correctness
        split_lengths = [len(split) for split in splits]
        return split_lengths
    
    
    
    