import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.hub.load)
class TorchHubLoadTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_hub_load_correctness(self):
        # Randomly choose between 'github' and 'local' source
        source = random.choice(['github', 'local'])
        
        if source == 'github':
            # Randomly select a repo owner and repo name
            repo_owner = random.choice(['pytorch', 'facebookresearch', 'huggingface'])
            repo_name = random.choice(['vision', 'fairseq', 'transformers'])
            ref = random.choice(['main', 'master', 'v0.10.0', 'v1.0.0'])
            repo_or_dir = f"{repo_owner}/{repo_name}:{ref}"
        else:
            # Randomly generate a local directory path
            repo_or_dir = f"/some/local/path/{random.choice(['pytorch', 'facebookresearch', 'huggingface'])}/{random.choice(['vision', 'fairseq', 'transformers'])}"
        
        # Randomly select a model name
        model_name = random.choice(['resnet50', 'bert-base-uncased', 'transformer'])
        
        # Load the model using torch.hub.load
        model = torch.hub.load(repo_or_dir, model_name, source=source)
        
        return model
    
    
    
    