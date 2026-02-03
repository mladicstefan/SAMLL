#include "nn.h"
#include "include/synclog.h"

void network_print(const network_t *nn);

network_t *network_create(u32 *sizes, u32 num_layers);
void network_randomize_weights(network_t *nn);
void network_destroy(network_t *nn);

void forward(network_t *nn, matrix_t *input);
void backward(network_t *nn, const matrix_t *expected);
void update_weights(network_t *nn, const f64 LEARNING_RATE);

void train(network_t *nn, matrix_t **inputs, matrix_t **labels,
           const u32 num_samples, u32 epochs, const f64 LEARNING_RATE);
u32 predict(network_t *nn, const matrix_t *input);
f64 MSE_loss(const matrix_t *predicted, const matrix_t *expected);

int main()
{
    LOG_DBG("Work in progress...");
    return 0;
}
