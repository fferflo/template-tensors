import unittest, sys
import ${TEST_MODULE} as test_module
try:
    import numpy as np
    import tensorflow as tf
except:
    sys.exit(0)

class TestStore(unittest.TestCase):
    def test_store_and_load_cpu(self):
        array = np.random.rand(3, 2).astype("float32")
        storage = test_module.StorageCpu(array.shape[0], array.shape[1])

        with tf.device("CPU"):
            array = tf.convert_to_tensor(array)
            array = tf.identity(array)

            storage.store(array)
            loaded = storage.load()

            self.assertTrue(np.all(loaded.numpy() == array.numpy()))

    def test_store_and_load_gpu(self):
        if "StorageGpu" in vars(test_module):
            array = np.random.rand(3, 2).astype("float32")
            storage = test_module.StorageGpu(array.shape[0], array.shape[1])

            with tf.device("GPU"):
                array = tf.convert_to_tensor(array)
                array = tf.identity(array)
                storage.store(array)
                loaded = storage.load()

                self.assertTrue(np.all(loaded.numpy() == array.numpy()))

if __name__ == '__main__':
    unittest.main()
