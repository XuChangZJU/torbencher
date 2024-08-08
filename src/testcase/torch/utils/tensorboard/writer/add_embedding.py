import random

import torch
from torch.utils.tensorboard import SummaryWriter

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.utils.tensorboard.writer.add_embedding)
class TorchUtilsTensorboardWriterAddUembeddingTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_add_embedding_correctness(self):
        writer = SummaryWriter()

        # Randomly generate the number of embeddings and their dimensions
        num_embeddings = random.randint(1, 10)
        embedding_dim = random.randint(1, 10)

        # Generate random embeddings
        embeddings = torch.randn(num_embeddings, embedding_dim)

        # Generate random metadata
        metadata = [f"label_{i}" for i in range(num_embeddings)]

        # Generate random label images
        label_img_size = (num_embeddings, 3, 32, 32)  # Assuming 3 channels and 32x32 image size
        label_img = torch.randn(label_img_size)

        # Add embeddings to the writer
        writer.add_embedding(embeddings, metadata=metadata, label_img=label_img)

        # Close the writer
        writer.close()
