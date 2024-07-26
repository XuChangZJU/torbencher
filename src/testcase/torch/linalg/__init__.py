# from .svdvals import TorchLinalgSvdvalsTestCase
# from .multi_dot import TorchLinalgMultidotTestCase
# from .vander import TorchLinalgVanderTestCase
# from .eigh import TorchLinalgEighTestCase
# from .lu_factor_ex import TorchLinalgLufactorexTestCase
# from .eig import TorchLinalgEigTestCase
# from .cond import TorchLinalgCondTestCase
# from .vector_norm import TorchLinalgVectornormTestCase
# from .cross import TorchLinalgCrossTestCase
# from .cholesky_ex import TorchLinalgCholeskyexTestCase
# from .slogdet import TorchLinalgSlogdetTestCase
# from .solve_ex import TorchLinalgSolveexTestCase
# from .householder_product import TorchLinalgHouseholderproductTestCase
# from .matrix_rank import TorchLinalgMatrixrankTestCase
# from .qr import TorchLinalgQrTestCase
# from .matrix_exp import TorchLinalgMatrixexpTestCase
# from .lu_factor import TorchLinalgLufactorTestCase
# from .lu import TorchLinalgLuTestCase
# from .eigvals import TorchLinalgEigvalsTestCase
# from .lu_solve import TorchLinalgLusolveTestCase
# from .matmul import TorchLinalgMatmulTestCase
# from .det import TorchLinalgDetTestCase
# from .tensorinv import TorchLinalgTensorinvTestCase
# from .solve_triangular import TorchLinalgSolvetriangularTestCase
# from .inv import TorchLinalgInvTestCase
# from .ldl_factor_ex import TorchLinalgLdlfactorexTestCase
# from .tensorsolve import TorchLinalgTensorsolveTestCase
# from .lstsq import TorchLinalgLstsqTestCase
# from .inv_ex import TorchLinalgInvexTestCase
# from .vecdot import TorchLinalgVecdotTestCase
# from .diagonal import TorchLinalgDiagonalTestCase
# from .ldl_solve import TorchLinalgLdlsolveTestCase
# from .matrix_norm import TorchLinalgMatrixnormTestCase
# from .solve import TorchLinalgSolveTestCase
# from .norm import TorchLinalgNormTestCase
# from .eigvalsh import TorchLinalgEigvalshTestCase
# from .svd import TorchLinalgSvdTestCase
# from .pinv import TorchLinalgPinvTestCase
# from .matrix_power import TorchLinalgMatrixpowerTestCase
# from .ldl_factor import TorchLinalgLdlfactorTestCase
# from .cholesky import TorchLinalgCholeskyTestCase

import os
import importlib
import logging

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase

# 设置日志配置
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

current_directory: str = os.path.dirname(os.path.abspath(__file__))
script_files: list = [f for f in os.listdir(current_directory) if f.endswith('.py') and f != '__init__.py']

for script_file in script_files:
    module_name: str = script_file[:-3]  # Remove the .py extension
    try:
        module = importlib.import_module(f'.{module_name}', package=__package__)
        # logger.debug(f"Successfully imported module {module_name}")
    except Exception as e:
        logger.debug(f"Failed to import module {module_name}: {e}")
        continue

    for attribute_name in dir(module):
        attribute = getattr(module, attribute_name)
        if isinstance(attribute, type) and issubclass(attribute, TorBencherTestCaseBase):
            globals()[attribute_name] = attribute
