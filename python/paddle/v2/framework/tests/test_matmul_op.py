import unittest
import numpy as np
from op_test import OpTest

# TODO(mkliegl):
# - all transpose variants
# - gradients


class TestMatMulOpBase(OpTest):
    """Matrix-Matrix test"""

    def setUp(self):
        self.op_type = "matmul"
        X = np.random.random((3, 7)).astype("float32")
        Y = np.random.random((7, 4)).astype("float32")
        self.inputs = {'X': X, 'Y': Y}
        self.outputs = {'Out': np.matmul(X, Y)}

    def test_check_output(self):
        self.check_output(atol=1e-2)


#     def test_check_grad_normal(self):
#         self.check_grad(['X', 'Y'], 'Out', max_relative_error=0.5)
# 
#     def test_check_grad_ingore_x(self):
#         self.check_grad(
#             ['Y'], 'Out', max_relative_error=0.5, no_grad_set=set("X"))
# 
#     def test_check_grad_ingore_y(self):
#         self.check_grad(
#             ['X'], 'Out', max_relative_error=0.5, no_grad_set=set('Y'))


class TestMatMulOpVecVec(TestMatMulOpBase):
    """Vector-Vector test"""

    def setUp(self):
        self.op_type = "matmul"
        X = np.random.random((2, )).astype("float32")
        Y = np.random.random((2, )).astype("float32")
        self.inputs = {'X': X, 'Y': Y}
        self.outputs = {'Out': np.matmul(X, Y)}


class TestMatMulOpMatVec(TestMatMulOpBase):
    """Matrix-Vector test"""

    def setUp(self):
        self.op_type = "matmul"
        X = np.random.random((4, 3)).astype("float32")
        Y = np.random.random((3, )).astype("float32")
        self.inputs = {'X': X, 'Y': Y}
        self.outputs = {'Out': np.matmul(X, Y)}


class TestMatMulOpVecMat(TestMatMulOpBase):
    """Vector-Matrix test"""

    def setUp(self):
        self.op_type = "matmul"
        X = np.random.random((2, )).astype("float32")
        Y = np.random.random((2, 3)).astype("float32")
        self.inputs = {'X': X, 'Y': Y}
        self.outputs = {'Out': np.matmul(X, Y)}


class TestMatMulOpVecBatchedMat(TestMatMulOpBase):
    """Vector-Batched Matrix test"""

    def setUp(self):
        self.op_type = "matmul"
        X = np.random.random((2, )).astype("float32")
        Y = np.random.random((3, 2, 5)).astype("float32")
        self.inputs = {'X': X, 'Y': Y}
        self.outputs = {'Out': np.matmul(X, Y)}


class TestMatMulOpBatchedMatVec(TestMatMulOpBase):
    """Batched Matrix-Vector test"""

    def setUp(self):
        self.op_type = "matmul"
        X = np.random.random((4, 2, 3)).astype("float32")
        Y = np.random.random((3, )).astype("float32")
        self.inputs = {'X': X, 'Y': Y}
        self.outputs = {'Out': np.matmul(X, Y)}


class TestMatMulOpBatchedMatMat(TestMatMulOpBase):
    """Batched Matrix-Matrix test"""

    def setUp(self):
        self.op_type = "matmul"
        X = np.random.random((4, 2, 3)).astype("float32")
        Y = np.random.random((3, 5)).astype("float32")
        self.inputs = {'X': X, 'Y': Y}
        self.outputs = {'Out': np.matmul(X, Y)}


class TestMatMulOpMatBatchedMat(TestMatMulOpBase):
    """Matrix-Batched Matrix test"""

    def setUp(self):
        self.op_type = "matmul"
        X = np.random.random((2, 3)).astype("float32")
        Y = np.random.random((4, 3, 5)).astype("float32")
        self.inputs = {'X': X, 'Y': Y}
        self.outputs = {'Out': np.matmul(X, Y)}


class TestMatMulOpBatchedMatBatchedMat(TestMatMulOpBase):
    """Batched Matrix-Batched Matrix test"""

    def setUp(self):
        self.op_type = "matmul"
        X = np.random.random((7, 3, 5)).astype("float32")
        Y = np.random.random((7, 5, 4)).astype("float32")
        self.inputs = {'X': X, 'Y': Y}
        self.outputs = {'Out': np.matmul(X, Y)}


if __name__ == "__main__":
    unittest.main()
