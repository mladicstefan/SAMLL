#include "nn.h"
#include "activations.h"
#include "include/random.h"
#include "include/synclog.h"
#include <memory.h>
#include <stdlib.h>

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

    nn->dW = calloc(num_layers, sizeof(matrix_t *));
    nn->db = calloc(num_layers, sizeof(matrix_t *));
    nn->delta = calloc(num_layers, sizeof(matrix_t *));

    for (u32 i = 0; i < num_layers - 1; i++)
    {
        nn->dW[i] = mat_init(nn->sizes[i + 1], nn->sizes[i]);
        nn->db[i] = mat_init(nn->sizes[i + 1], 1);
    }

    for (u32 i = 0; i < num_layers; i++)
    {
        nn->delta[i] = mat_init(nn->sizes[i], 1);
    }

    network_randomize_weights(nn);

    return nn;
}

void network_randomize_weights(network_t *nn)
{
    for (u32 i = 0; i < nn->num_layers - 1; i++)
    {
        // Xavier initialization: scale by sqrt(1/n_in)
        f64 scale = sqrt(1.0 / nn->sizes[i]);
        matrix_fill_rand(nn->weights[i]);
        mat_scalar_mult(nn->weights[i], scale);

        matrix_fill_rand(nn->biases[i]);
        mat_scalar_mult(nn->biases[i], 0.01);
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
    for (u32 i = 0; i < nn->num_layers; i++)
    {
        mat_destroy(nn->dW[i]);
        mat_destroy(nn->db[i]);
        mat_destroy(nn->delta[i]);
    }

    free(nn->dW);
    free(nn->db);
    free(nn->delta);
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

static void compute_output_layer_delta(network_t *nn, const matrix_t *expected)
{
    u32 L = nn->num_layers - 1;

    matrix_t *sub_result = mat_sub(nn->a[L], expected);
    matrix_t *sig_deriv = mat_sigmoid_derivative(nn->z[L]);
    matrix_t *delta_L = mat_hadamard(sub_result, sig_deriv);

    u32 n = delta_L->rows * delta_L->cols;
    memcpy(nn->delta[L]->data, delta_L->data, n * sizeof(f64));

    mat_destroy(sub_result);
    mat_destroy(sig_deriv);
    mat_destroy(delta_L);
}

static void compute_hidden_layers_delta(network_t *nn)
{
    for (u32 L = nn->num_layers - 2; L >= 1; L--)
    {
        matrix_t *tW = mat_transpose(nn->weights[L]);
        matrix_t *dL = nn->delta[L + 1];
        matrix_t *product = mat_mult(tW, dL);
        matrix_t *activation_derivative = mat_reLU_derivative(nn->z[L]);

        matrix_t *res = mat_hadamard(product, activation_derivative);
        memcpy(nn->delta[L]->data, res->data,
               res->rows * res->cols * sizeof(f64));

        mat_destroy(tW);
        mat_destroy(product);
        mat_destroy(activation_derivative);
        mat_destroy(res);
    }
}

static void compute_weight_gradients(network_t *nn)
{
    for (u32 l = 0; l < nn->num_layers - 1; l++)
    {
        matrix_t *aT = mat_transpose(nn->a[l]);
        matrix_t *res = mat_mult(nn->delta[l + 1], aT);

        memcpy(nn->dW[l]->data, res->data, res->rows * res->cols * sizeof(f64));

        mat_destroy(res);
        mat_destroy(aT);
    }
}
static void compute_bias_gradients(network_t *nn)
{
    for (u32 l = 0; l < nn->num_layers - 1; l++)
    {
        matrix_t *res = nn->delta[l + 1];
        memcpy(nn->db[l]->data, res->data, res->rows * res->cols * sizeof(f64));
    }
}

void backward(network_t *nn, const matrix_t *expected)
{
    compute_output_layer_delta(nn, expected);
    compute_hidden_layers_delta(nn);
    compute_weight_gradients(nn);
    compute_bias_gradients(nn);
}

void update_parameters(network_t *nn, const f64 LEARNING_RATE)
{
    for (u32 l = 0; l < nn->num_layers - 1; l++)
    {
        for (u32 i = 0; i < nn->weights[l]->rows * nn->weights[l]->cols; i++)
            nn->weights[l]->data[i] -= LEARNING_RATE * nn->dW[l]->data[i];

        for (u32 i = 0; i < nn->biases[l]->rows * nn->biases[l]->cols; i++)
            nn->biases[l]->data[i] -= LEARNING_RATE * nn->db[l]->data[i];
    }
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
            update_parameters(nn, LEARNING_RATE);
        }

        if (epoch % 10000 == 0 || epoch == 0)
        {
            LOG_DBG("Epoch %lu: Avg Loss = %.6f", epoch,
                    total_loss / num_samples);
        }
    }
}

u32 predict(network_t *nn, const matrix_t *input)
{
    memcpy(nn->a[0]->data, input->data,
           input->rows * input->cols * sizeof(f64));
    forward(nn);

    u32 max_idx = 0;
    f64 max_val = nn->a[nn->num_layers - 1]->data[0];
    for (u32 i = 1; i < 3; i++)
    {
        if (nn->a[nn->num_layers - 1]->data[i] > max_val)
        {
            max_val = nn->a[nn->num_layers - 1]->data[i];
            max_idx = i;
        }
    }
    return max_idx;
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
    train(nn, data, 150, 100000, 0.01);
    u32 correct = 0;
    for (u32 i = 0; i < 150; i++)
    {
        u32 predicted = predict(nn, data->inputs[i]);

        u32 actual = 0;
        for (u32 j = 0; j < 3; j++)
        {
            if (data->labels[i]->data[j] == 1.0)
            {
                actual = j;
                break;
            }
        }

        if (predicted == actual)
            correct++;

        LOG_DBG("Sample %u: Predicted=%u, Actual=%u %s", i, predicted, actual,
                (predicted == actual) ? "✓" : "✗");
    }

    LOG_DBG("Accuracy: %u/%u = %.2f%%", correct, 150, (100.0 * correct) / 150);
    dataset_destroy(data);
    network_destroy(nn);
    return 0;
}
