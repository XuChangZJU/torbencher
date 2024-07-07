import torch
import random
from torch.utils.tensorboard import SummaryWriter


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.utils.tensorboard.writer.add_mesh)
class TorchUtilsTensorboardWriterAddmeshTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_add_mesh_correctness(self):
        writer = SummaryWriter()
    
        # Randomly generate the number of vertices and faces
        num_vertices = random.randint(3, 10)  # Minimum 3 vertices to form a face
        num_faces = random.randint(1, 10)
    
        # Randomly generate vertices and faces
        vertices = torch.randn(num_vertices, 3)  # 3D vertices
        faces = torch.randint(0, num_vertices, (num_faces, 3), dtype=torch.int32)  # Faces are defined by indices of vertices
    
        # Randomly generate colors if needed
        colors = torch.randn(num_vertices, 3)  # RGB colors for each vertex
    
        # Add mesh to the writer
        writer.add_mesh('random_mesh', vertices, faces, colors)
    
        # Close the writer
        writer.close()
    
    
    
    