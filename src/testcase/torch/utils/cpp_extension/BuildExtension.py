import torch
import random
from torch.utils.cpp_extension import BuildExtension, CppExtension

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.utils.cpp_extension.BuildExtension)
class TorchUtilsCppextensionBuildextensionTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_build_extension_correctness(self):
        # Randomly generate a name for the extension
        extension_name = f"extension_{random.randint(1, 1000)}"

        # Randomly generate a list of source files (for simplicity, using dummy file names)
        num_sources = random.randint(1, 3)
        sources = [f"source_{i}.cpp" for i in range(num_sources)]

        # Randomly generate a list of include directories (for simplicity, using dummy directory names)
        num_includes = random.randint(1, 3)
        include_dirs = [f"include_dir_{i}" for i in range(num_includes)]

        # Randomly generate a list of library directories (for simplicity, using dummy directory names)
        num_library_dirs = random.randint(1, 3)
        library_dirs = [f"library_dir_{i}" for i in range(num_library_dirs)]

        # Randomly generate a list of libraries (for simplicity, using dummy library names)
        num_libraries = random.randint(1, 3)
        libraries = [f"library_{i}" for i in range(num_libraries)]

        # Create the CppExtension object with the generated parameters
        cpp_extension = CppExtension(
            name=extension_name,
            sources=sources,
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=libraries
        )

        # Create the BuildExtension object
        build_extension = BuildExtension()

        return cpp_extension, build_extension
