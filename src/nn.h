#pragma once
#include "dataset.h"
#include "include/types.h"
#include "matrix.h"

typedef struct
{
    u32 num_layers;
    u32 *sizes;
    matrix_t **weights;
    matrix_t **biases;
    matrix_t **z; // pre-activation
    matrix_t **a; // post-activation
} network_t;

void network_print(const network_t *nn);

network_t *network_create(const u32 *sizes, const u32 num_layers);
void network_randomize_weights(network_t *nn);
void network_destroy(network_t *nn);

void forward(network_t *nn);
void backward(network_t *nn, const matrix_t *expected);
void update_weights(network_t *nn, const f64 LEARNING_RATE);

void train(network_t *nn, dataset_t *data, const u32 num_samples, u32 epochs,
           const f64 LEARNING_RATE);
u32 predict(network_t *nn, const matrix_t *input);
f64 MSE_loss(const matrix_t *predicted, const matrix_t *expected);
matrix_t *mat_sigmoid(matrix_t *m);
matrix_t *mat_reLU(matrix_t *m);
