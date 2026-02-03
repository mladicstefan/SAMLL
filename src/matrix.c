#include "include/random.h"
#include "include/synclog.h"
#include "include/types.h"

#include <stdint.h>
#include <stdlib.h>

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

static void DBG_PRINT_MATRIX(matrix_t *m)
{
    LOG_DBG("Matrix (%u x %u):", m->rows, m->cols);
    LOG_DBG("At Memory address: %p", (void *)m);
    for (u32 i = 0; i < m->rows; i++)
    {
        fprintf(stderr, "[");
        for (u32 j = 0; j < m->cols; j++)
        {
            fprintf(stderr, "%8.3f", m->data[i * m->cols + j]);
            if (j < m->cols - 1)
                fprintf(stderr, ", ");
        }
        fprintf(stderr, "]\n");
    }
}

matrix_t *mat_init(u32 rows, u32 cols)
{
    matrix_t *m = malloc(sizeof(matrix_t));
    if (!m)
        return NULL;

    m->rows = rows;
    m->cols = cols;
    m->data = calloc(rows * cols, sizeof(f64));

    if (!m->data)
    {
        free(m);
        return NULL;
    }

    return m;
}

void mat_destroy(matrix_t *m)
{
    if (m)
    {
        free(m->data);
        free(m);
    }
}

matrix_t *mat_add(matrix_t *a, matrix_t *b)
{
    if (a->cols != b->cols || a->rows != b->rows)
    {
        LOG_DBG("Mismatch in dimensions a:(%u x %u) b:(%u x %u)", a->rows,
                a->cols, b->rows, b->cols);
        return NULL;
    }

    matrix_t *res = mat_init(a->rows, a->cols);
    u32 total = res->rows * res->cols;

    for (u32 i = 0; i < total; i++)
        res->data[i] = a->data[i] + b->data[i];
    return res;
}

matrix_t *mat_sub(matrix_t *a, matrix_t *b)
{
    if (a->cols != b->cols || a->rows != b->rows)
    {
        LOG_DBG("Mismatch in dimensions a:(%u x %u) b:(%u x %u)", a->rows,
                a->cols, b->rows, b->cols);
        return NULL;
    }

    matrix_t *res = mat_init(a->rows, a->cols);
    u32 total = res->rows * res->cols;

    for (u32 i = 0; i < total; i++)
        res->data[i] = a->data[i] - b->data[i];
    return res;
}

void mat_scalar_mult(matrix_t *m, f64 k)
{
    if (!m)
    {
        LOG_DBG("NULL PTR: %p", (void *)m);
        return;
    }

    for (u32 i = 0; i < m->rows * m->cols; i++)
        m->data[i] *= k;
}

void matrix_fill_rand(matrix_t *m)
{
    if (!m)
    {
        LOG_DBG("NULL PTR EXCEPTION: %p", (void *)m);
        return;
    }

    for (u32 i = 0; i < m->rows * m->cols; i++)
    {
        m->data[i] += random_f64();
    }
}

matrix_t *mat_transpose(matrix_t *m)
{
    if (!m)
    {
        LOG_DBG("NULL PTR: %p", (void *)m);
        return NULL;
    }

    // transposed
    matrix_t *t = mat_init(m->cols, m->rows);

    for (u32 i = 0; i < m->rows; i++)
    {
        for (u32 j = 0; j < m->cols; j++)
        {
            t->data[j * t->cols + i] = m->data[i * m->cols + j];
        }
    }
    return t;
}

/*
 * Cache-optimized mat mult using blocking
 * Instead of iterating through entire rows/columns, process 32x32 blocks
 * at a time. This keeps the working data in L1 cache , dramatically
 * reducing cache misses
 * */
matrix_t *mat_mult(matrix_t *a, matrix_t *b)
{
    matrix_t *res = mat_init(a->rows, b->cols);

    const u32 BLOCK = 32;

    for (u32 ii = 0; ii < a->rows; ii += BLOCK)
    {
        for (u32 jj = 0; jj < b->cols; jj += BLOCK)
        {
            for (u32 kk = 0; kk < a->cols; kk += BLOCK)
            {
                u32 i_end = (ii + BLOCK < a->rows) ? ii + BLOCK : a->rows;
                u32 j_end = (jj + BLOCK < b->cols) ? jj + BLOCK : b->cols;
                u32 k_end = (kk + BLOCK < a->cols) ? kk + BLOCK : a->cols;

                for (u32 i = ii; i < i_end; i++)
                {
                    for (u32 k = kk; k < k_end; k++)
                    {
                        f64 aik = a->data[i * a->cols + k];
                        for (u32 j = jj; j < j_end; j++)
                        {
                            res->data[i * res->cols + j] +=
                                aik * b->data[k * b->cols + j];
                        }
                    }
                }
            }
        }
    }
    return res;
}

// f64 benchmark(mat_mult_fn fn, matrix_t *a, matrix_t *b, i64 runs)
// {
//     struct timespec start, end;
//
//     clock_gettime(CLOCK_MONOTONIC, &start);
//     for (int i = 0; i < runs; i++)
//     {
//         matrix_t *res = fn(a, b);
//         mat_destroy(res);
//     }
//     clock_gettime(CLOCK_MONOTONIC, &end);
//
//     double elapsed =
//         (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
//     return elapsed / runs;
// }

// int main()
// {
//     u8 rows = 4, cols = 4;
//     matrix_t *a = mat_init(rows, cols);
//
//     for (u32 i = 0; i < rows * cols; i++)
//         a->data[i] = i;
//     matrix_t *b = mat_init(rows, cols);
//
//     for (u32 i = 0; i < rows * cols; i++)
//         b->data[i] = i;
//     matrix_t *res = mat_add(a, b);
//
//     DBG_PRINT_MATRIX(res);
//     matrix_t *t = mat_transpose(res);
//     DBG_PRINT_MATRIX(t);
//     matrix_t *mul = mat_mult(a, b);
//     DBG_PRINT_MATRIX(mul);
//
//     mat_destroy(mul);
//     mat_destroy(a);
//     mat_destroy(b);
//     mat_destroy(res);
//     mat_destroy(t);
//     return 0;
// }
