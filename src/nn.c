#include "nn.h"
#include "activations.h"
#include "include/random.h"
#include "include/synclog.h"
#include <memory.h>
#include <stdlib.h>

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

void network_print(const network_t *nn)
{
    for (u32 i = 0; i < nn->num_layers - 1; i++)
    {
        LOG_DBG("Weight %u", i);
        DBG_PRINT_MATRIX(nn->weights[i]);
        LOG_DBG("Bias %u", i);
        DBG_PRINT_MATRIX(nn->biases[i]);
    }

    for (u32 i = 0; i < nn->num_layers; i++)
    {
        if (i == 0)
        {
            LOG_DBG("INPUTS: %d", i);
            DBG_PRINT_MATRIX(nn->a[i]);
            continue;
        }
        LOG_DBG("Neuron (pre-activation) %u", i);
        DBG_PRINT_MATRIX(nn->z[i]);
        LOG_DBG("Neuron (post-activation) %u", i);
        DBG_PRINT_MATRIX(nn->a[i]);
    }
}

network_t *network_create(const u32 *sizes, const u32 num_layers)
{
    network_t *nn = malloc(sizeof(network_t));
    if (!nn)
    {
        LOG_DBG("Malloc failed, network not allocated %p", (void *)nn);
        return NULL;
    }

    nn->num_layers = num_layers;
    nn->sizes = calloc(num_layers, sizeof(u32));
    for (u32 i = 0; i < num_layers; i++)
    {
        nn->sizes[i] = sizes[i];
    }

    nn->weights = calloc(num_layers, sizeof(matrix_t *));
    nn->biases = calloc(num_layers, sizeof(matrix_t *));
    for (u32 i = 0; i < num_layers - 1; i++)
    {
        nn->weights[i] = mat_init(nn->sizes[i + 1], nn->sizes[i]);
        nn->biases[i] = mat_init(nn->sizes[i + 1], 1);
    }

    nn->z = calloc(num_layers, sizeof(matrix_t *));
    nn->a = calloc(num_layers, sizeof(matrix_t *));

    for (u32 i = 0; i < num_layers; i++)
    {
        nn->z[i] = mat_init(nn->sizes[i], 1);
        nn->a[i] = mat_init(nn->sizes[i], 1);
    }

    network_randomize_weights(nn);

    return nn;
}
void network_randomize_weights(network_t *nn)
{
    for (u32 i = 0; i < nn->num_layers - 1; i++)
    {
        matrix_fill_rand(nn->weights[i]);
        matrix_fill_rand(nn->biases[i]);
    }
}
void network_destroy(network_t *nn)
{
    if (!nn)
    {
        LOG_DBG("Error, *nn already %p", (void *)nn);
        return;
    }
    for (u32 i = 0; i < nn->num_layers - 1; i++)
    {
        mat_destroy(nn->weights[i]);
        mat_destroy(nn->biases[i]);
    }
    free(nn->weights);
    free(nn->biases);
    free(nn->sizes);

    for (u32 i = 0; i < nn->num_layers; i++)
    {
        mat_destroy(nn->z[i]);
        mat_destroy(nn->a[i]);
    }
    free(nn->a);
    free(nn->z);
    free(nn);
}

void forward(network_t *nn)
{
    for (u32 i = 0; i < nn->num_layers - 1; i++)
    {
        matrix_t *mult_result = mat_mult(nn->weights[i], nn->a[i]);
        matrix_t *new_z = mat_add(mult_result, nn->biases[i]);
        mat_destroy(mult_result);

        mat_destroy(nn->z[i + 1]);
        nn->z[i + 1] = new_z;

        matrix_t *new_a = (i == nn->num_layers - 2) ? mat_sigmoid(nn->z[i + 1])
                                                    : mat_reLU(nn->z[i + 1]);
        mat_destroy(nn->a[i + 1]);
        nn->a[i + 1] = new_a;
    }
}

void backward(network_t *nn, const matrix_t *expected)
{
    (void)nn;
    (void)expected;
    // TODO: implement
}

void update_weights(network_t *nn, const f64 LEARNING_RATE)
{
    (void)nn;
    (void)LEARNING_RATE;
    // TODO: implement
}

void train(network_t *nn, dataset_t *data, const u32 num_samples, u32 epochs,
           const f64 LEARNING_RATE)
{

    for (u64 epoch = 0; epoch < epochs; epoch++)
    {
        f64 total_loss = 0.0;
        for (u32 i = 0; i < num_samples; i++)
        {
            u32 size = nn->a[0]->rows * nn->a[0]->cols * sizeof(f64);
            memcpy(nn->a[0]->data, data->inputs[i]->data, size);
            forward(nn);

            total_loss += MSE_loss(nn->a[nn->num_layers - 1], data->labels[i]);

            backward(nn, data->labels[i]);
            update_weights(nn, LEARNING_RATE);
        }

        if (epoch % 10 == 0)
        {
            LOG_DBG("Epoch %lu: Avg Loss = %.6f", epoch,
                    total_loss / num_samples);
        }
    }
}

u32 predict(network_t *nn, const matrix_t *input)
{
    (void)nn;
    (void)input;
    return 0;
}

// MSE = (1/n) Σ(predicted - expected)²
f64 MSE_loss(const matrix_t *predicted, const matrix_t *expected)
{
    if (!predicted || !expected)
    {
        LOG_DBG("Invalid inputs: %p %p", (void *)predicted, (void *)expected);
        return -1;
    }
    f64 loss = 0;

    matrix_t *res = mat_sub(predicted, expected);
    for (u32 i = 0; i < res->rows * res->cols; i++)
    {
        res->data[i] = pow(res->data[i], 2);
        loss += res->data[i];
    }
    u32 n = res->rows * res->cols;
    mat_destroy(res);
    return loss / n;
}

int main()
{
    rng_seed();
    const u32 SIZES[3] = {4, 8, 3};
    const u32 NUM_LAYERS = 3;
    network_t *nn = network_create(SIZES, NUM_LAYERS);
    network_print(nn);
    dataset_t *data = parse_file("../data/iris/iris.data");
    // dataset_print(data);
    train(nn, data, 150, 10000, 0.5);
    dataset_destroy(data);
    network_destroy(nn);
    return 0;
}
