/*iris dataset parser*/
#include "dataset.h"
#include "include/synclog.h"
#include "include/types.h"
#include "string.h"
#include <stdio.h>
static const u8 BUF_LEN = 255;
static const u8 NUM_FEATURES = 4;

static dataset_t *dataset_create(u32 num_samples)
{
    dataset_t *dataset = malloc(sizeof(dataset_t));
    dataset->inputs = calloc(num_samples, sizeof(matrix_t *));
    dataset->labels = calloc(num_samples, sizeof(matrix_t *));
    dataset->num_samples = num_samples;

    return dataset;
}

dataset_t *parse_file(char *path)
{
    FILE *f = fopen(path, "r");
    if (!f)
    {
        LOG_DBG("Error cannot open file at %s : %p", path, (void *)f);
        return NULL;
    }

    dataset_t *dataset = dataset_create(150);
    char buf[BUF_LEN];
    u32 idx = 0;
    while (fgets(buf, sizeof(buf), f) && idx < 150)
    {
        dataset->inputs[idx] = mat_init(NUM_FEATURES, 1);
        for (u8 i = 0; i < NUM_FEATURES; i++)
        {
            if (i == 0)
                dataset->inputs[idx]->data[i] = atof(strtok(buf, ","));
            else
                dataset->inputs[idx]->data[i] = atof(strtok(NULL, ","));
        }
        dataset->labels[idx] = mat_init(3, 1);
        char *label = strtok(NULL, "\n");
        if (strstr(label, "setosa"))
            dataset->labels[idx]->data[0] = 1.0;
        else if (strstr(label, "versicolor"))
            dataset->labels[idx]->data[1] = 1.0;
        else if (strstr(label, "virginica"))
            dataset->labels[idx]->data[2] = 1.0;
        idx++;
    }

    fclose(f);
    return dataset;
}

void dataset_print(const dataset_t *data)
{
    LOG_DBG("Dataset: %u samples", data->num_samples);
    LOG_DBG("Input size: %ux%u", data->inputs[0]->rows, data->inputs[0]->cols);
    LOG_DBG("Label size: %ux%u", data->labels[0]->rows, data->labels[0]->cols);
    LOG_DBG("-----------------------------");

    for (u32 i = 0; i < data->num_samples; i++)
    {
        LOG_DBG("Sample %u:", i);
        LOG_DBG("  Input:  [%.2f, %.2f, %.2f, %.2f]", data->inputs[i]->data[0],
                data->inputs[i]->data[1], data->inputs[i]->data[2],
                data->inputs[i]->data[3]);
        LOG_DBG("  Label:  [%.1f, %.1f, %.1f] (%s)", data->labels[i]->data[0],
                data->labels[i]->data[1], data->labels[i]->data[2],
                data->labels[i]->data[0] == 1.0   ? "setosa"
                : data->labels[i]->data[1] == 1.0 ? "versicolor"
                                                  : "virginica");
    }
}

void dataset_print_n(const dataset_t *data, u32 n)
{
    if (n > data->num_samples)
        n = data->num_samples;

    LOG_DBG("Dataset: %u samples (showing %u)", data->num_samples, n);
    LOG_DBG("-----------------------------");

    for (u32 i = 0; i < n; i++)
    {
        LOG_DBG("[%3u] In:[%5.2f %5.2f %5.2f %5.2f] Out:[%.0f %.0f %.0f]", i,
                data->inputs[i]->data[0], data->inputs[i]->data[1],
                data->inputs[i]->data[2], data->inputs[i]->data[3],
                data->labels[i]->data[0], data->labels[i]->data[1],
                data->labels[i]->data[2]);
    }
}

void dataset_destroy(dataset_t *data)
{
    if (!data)
        return;
    for (u32 i = 0; i < data->num_samples; i++)
    {
        mat_destroy(data->inputs[i]);
        mat_destroy(data->labels[i]);
    }
    free(data->inputs);
    free(data->labels);
    free(data);
}

// int main()
// {
//     dataset_t *data = parse_file("../data/iris/bezdekIris.data");
//     dataset_print(data);
//     dataset_destroy(data);
//     return 0;
// }
