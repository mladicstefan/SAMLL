from cffi import FFI

ffi = FFI()

ffi.cdef(
    """
    typedef struct {
        uint32_t rows, cols;
        double *data;
    } matrix_t;

    matrix_t *mat_init(uint32_t rows, uint32_t cols);
    void mat_destroy(matrix_t *m);
    matrix_t *mat_add(matrix_t *a, matrix_t *b);
    matrix_t *mat_sub(matrix_t *a, matrix_t *b);
    matrix_t *mat_mult(matrix_t *a, matrix_t *b);
    void mat_scalar_mult(matrix_t *m, double k);
    matrix_t *mat_transpose(matrix_t *m);
"""
)

lib = ffi.dlopen("../shared/libmatrix.so")


class Matrix:
    __slots__ = ("_ptr",)

    def __init__(self, rows, cols, data=None):
        self._ptr = lib.mat_init(rows, cols)
        if self._ptr == ffi.NULL:
            raise MemoryError("Failed to allocate matrix")

        if data:
            for i, val in enumerate(data):
                self._ptr.data[i] = float(val)

    def __del__(self):
        if getattr(self, "_ptr", ffi.NULL) != ffi.NULL:
            lib.mat_destroy(self._ptr)

    @classmethod
    def _from_ptr(cls, ptr):
        if ptr == ffi.NULL:
            raise ValueError("Matrix operation failed (NULL result)")
        obj = object.__new__(cls)
        obj._ptr = ptr
        return obj

    @property
    def shape(self):
        return (self._ptr.rows, self._ptr.cols)

    def __repr__(self):
        rows, cols = self.shape
        lines = []
        for i in range(rows):
            row = [self._ptr.data[i * cols + j] for j in range(cols)]
            lines.append("[" + ", ".join(f"{v:8.3f}" for v in row) + "]")
        return f"Matrix({rows}x{cols}):\n" + "\n".join(lines)

    def _check_type(self, other):
        if not isinstance(other, Matrix):
            return NotImplemented
        return True

    def __add__(self, other):
        if self._check_type(other) is NotImplemented:
            return NotImplemented
        return Matrix._from_ptr(lib.mat_add(self._ptr, other._ptr))

    def __sub__(self, other):
        if self._check_type(other) is NotImplemented:
            return NotImplemented
        return Matrix._from_ptr(lib.mat_sub(self._ptr, other._ptr))

    def __matmul__(self, other):
        if self._check_type(other) is NotImplemented:
            return NotImplemented
        return Matrix._from_ptr(lib.mat_mult(self._ptr, other._ptr))

    def __mul__(self, scalar):
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        result = Matrix._from_ptr(lib.mat_init(self._ptr.rows, self._ptr.cols))
        for i in range(self._ptr.rows * self._ptr.cols):
            result._ptr.data[i] = self._ptr.data[i]
        lib.mat_scalar_mult(result._ptr, float(scalar))
        return result

    __rmul__ = __mul__

    @property
    def T(self):
        return Matrix._from_ptr(lib.mat_transpose(self._ptr))
