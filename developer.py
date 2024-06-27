import unittest

# Import the test case
from src.testcase.torch.nn.ConvTranspose3d import TorchNnConvtranspose3dTestCase

def collect_test_results(test_case_class):
    """
    运行单个测试用例并收集结果
    """
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(test_case_class))
    runner = unittest.TextTestRunner(stream=None, verbosity=0)  # 静默模式
    result = runner.run(suite)
    return result.errors or result.failures  # 直接返回错误或失败列表，如果都不存在则返回False

def check_for_issues(test_case_class, iteration_count):
    """
    运行指定次数的测试并检查是否存在错误或失败
    """
    has_issues = False
    for _ in range(iteration_count):
        # 只需检查是否有错误或失败，不需要详细信息
        if collect_test_results(test_case_class):
            has_issues = True
            break

    if has_issues:
        print("Have false")
    else:
        print("All iterations completed successfully without any failures or errors.")

if __name__ == '__main__':
    check_for_issues(TorchNnConvtranspose3dTestCase, 1000)
