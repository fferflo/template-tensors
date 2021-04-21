import unittest, sys
import ${TEST_MODULE} as test_module
try:
    import numpy as np
except:
    sys.exit(0)

class TestStore(unittest.TestCase):
    def test_store_and_load(self):
        array = np.random.rand(3, 2).astype("float32")

        storage = test_module.Storage(array.shape[0], array.shape[1])
        storage.store(array)
        self.assertTrue(np.all(storage.load() == array))

if __name__ == '__main__':
    unittest.main()
