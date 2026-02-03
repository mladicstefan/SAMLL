#pragma once
#include "include/types.h"

typedef struct matrix_t
{
    u32 rows, cols;
    f64 *data;
} matrix_t;

matrix_t *mat_init(u32 rows, u32 cols);
void mat_destroy(matrix_t *m);
void matrix_fill_rand(matrix_t *m);

matrix_t *mat_add(matrix_t *a, matrix_t *b);
matrix_t *mat_sub(matrix_t *a, matrix_t *b);
matrix_t *mat_mult(matrix_t *a, matrix_t *b);
void mat_scalar_mult(matrix_t *a, f64 k);

matrix_t *mat_transpose(matrix_t *a);

/*TODO:*/
matrix_t *mat_inverse(matrix_t *a, matrix_t *b);
