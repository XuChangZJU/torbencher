import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.hub.list)
class TorchHubListTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_hub_list_correctness(self):
        # Randomly select a repo owner and repo name from a predefined list
        repo_owners = ['pytorch', 'facebookresearch', 'huggingface']
        repo_names = ['vision', 'fairseq', 'transformers']
        repo_owner = random.choice(repo_owners)
        repo_name = random.choice(repo_names)
        
        # Randomly decide whether to include a ref (tag or branch)
        include_ref = random.choice([True, False])
        if include_ref:
            refs = ['main', 'master', 'v1.0', 'v2.0']
            ref = random.choice(refs)
            github = f"{repo_owner}/{repo_name}:{ref}"
        else:
            github = f"{repo_owner}/{repo_name}"
        
        # Randomly decide whether to force reload
        force_reload = random.choice([True, False])
        
        # Randomly decide whether to skip validation
        skip_validation = random.choice([True, False])
        
        # Randomly decide the trust_repo parameter
        trust_repo_options = [True, False, "check", None]
        trust_repo = random.choice(trust_repo_options)
        
        # Call torch.hub.list with the generated parameters
        result = torch.hub.list(github, force_reload, skip_validation, trust_repo)
        return result
    