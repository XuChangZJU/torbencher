import unittest;

class CustomTestResult(unittest.TextTestResult):
    """
    **description**
    Custom test result class that stores return values of test methods.

    **params**
    *args: Variable length argument list. **kwargs: Arbitrary keyword arguments.
    """
    def __init__(self, *args, **kwargs):
        """
        **description**
        Initializes the CustomTestResult instance.

        **params**
        *args: Variable length argument list. **kwargs: Arbitrary keyword arguments.

        **returns**
        None
        """
        super().__init__(*args, **kwargs);
        self.returnValues = {};

    def addSuccess(self, test):
        """
        **description**
        Adds a successful test case and stores its return value.

        **params**
        test: The test case instance.

        **returns**
        None
        """
        super().addSuccess(test);
        method = getattr(test, test._testMethodName);
        if callable(method):
            try:
                returnValue = method();
                test_case_name = f"{test.test_case_name}";
                self.returnValues[test_case_name] = returnValue;
            except Exception as e:
                pass;

    def getReturnValues(self):
        """
        **description**
        Retrieves the stored return values of test cases.

        **params**
        None

        **returns**
        dict: A dictionary of test case names and their return values.
        """
        return self.returnValues;

class MyTestRunner(unittest.TextTestRunner):
    """
    **description**
    Custom test runner that uses CustomTestResult.

    **params**
    None
    """
    resultclass = CustomTestResult;

class MyTestLoader(unittest.TestLoader):
    """
    **description**
    Custom test loader that sets test_case_name for each test case.

    **params**
    None
    """
    def loadTestsFromTestCase(self, testCaseClass):
        """
        **description**
        Returns a suite of all test cases contained in testCaseClass.

        **params**
        testCaseClass (type): The test case class.

        **returns**
        unittest.TestSuite: A test suite of all test cases.

        **raises**
        TypeError: If testCaseClass is not a subclass of unittest.TestCase.
        """
        if issubclass(testCaseClass, unittest.TestCase):
            testCaseNames = self.getTestCaseNames(testCaseClass);
            if not testCaseNames and hasattr(testCaseClass, 'runTest'):
                testCaseNames = ['runTest'];
            test_cases = [testCaseClass(testCaseName) for testCaseName in testCaseNames];
            for test_case in test_cases:
                test_case.test_case_name = testCaseClass.__name__;
            return self.suiteClass(test_cases);
        raise TypeError("testCaseClass must be a subclass of unittest.TestCase");
