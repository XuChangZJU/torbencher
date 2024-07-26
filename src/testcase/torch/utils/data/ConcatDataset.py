import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.utils.data.ConcatDataset)
class TorchUtilsDataConcatdatasetTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_concatdataset_correctness(self):
        # Generate random parameters for the datasets
        num_datasets = random.randint(2, 5)  # Number of datasets to concatenate
        dataset_lengths = [random.randint(1, 10) for _ in range(num_datasets)]  # Length of each dataset
    
        # Create a list of random datasets
        datasets = []
        for length in dataset_lengths:
            dataset = [(torch.randn(random.randint(1, 10), random.randint(1, 10)), torch.randint(0, 10, (1,))) for _ in range(length)]
            datasets.append(dataset)
    
        # Create a ConcatDataset
        concat_dataset = torch.utils.data.ConcatDataset(datasets)
    
        # Get a random index within the concatenated dataset
        index = random.randint(0, len(concat_dataset) - 1)
    
        # Get the item at the random index
        result = concat_dataset[index]
        
        return result
    