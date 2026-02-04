#pragma once

#include "include/types.h"
#include "matrix.h"
#include <math.h>

f64 reLU(f64 z);
f64 sigmoid(f64 z);

f64 reLU(f64 z) { return (z <= 0) ? 0 : z; }

f64 reLU_derivative(f64 x) { return x > 0 ? 1.0 : 0.0; }

f64 sigmoid(f64 z) { return 1 / (1 + exp(-z)); }

f64 sigmoid_derivative(f64 sigmoid_output)
{
    return sigmoid_output * (1.0 - sigmoid_output);
}

matrix_t *mat_sigmoid(matrix_t *m)
{
    if (!m)
    {
        LOG_DBG("Invalid matrix: %p", (void *)m);
        return NULL;
    }

    matrix_t *res = mat_init(m->rows, m->cols);

    for (u32 i = 0; i < m->rows * m->cols; i++)
    {
        res->data[i] = sigmoid(m->data[i]);
    }

    return res;
}

matrix_t *mat_reLU(matrix_t *m)
{
    if (!m)
    {
        LOG_DBG("Invalid matrix: %p", (void *)m);
        return NULL;
    }

    matrix_t *res = mat_init(m->rows, m->cols);

    for (u32 i = 0; i < m->rows * m->cols; i++)
    {
        res->data[i] = reLU(m->data[i]);
    }

    return res;
}

matrix_t *mat_sigmoid_derivative(matrix_t *m)
{
    if (!m)
    {
        LOG_DBG("Invalid matrix: %p", (void *)m);
        return NULL;
    }

    matrix_t *res = mat_init(m->rows, m->cols);

    for (u32 i = 0; i < m->rows * m->cols; i++)
    {
        res->data[i] = sigmoid_derivative(sigmoid(m->data[i]));
    }

    return res;
}

matrix_t *mat_reLU_derivative(matrix_t *m)
{
    if (!m)
    {
        LOG_DBG("Invalid matrix: %p", (void *)m);
        return NULL;
    }

    matrix_t *res = mat_init(m->rows, m->cols);

    for (u32 i = 0; i < m->rows * m->cols; i++)
    {
        res->data[i] = reLU_derivative(m->data[i]);
    }

    return res;
}
