#pragma once
#ifndef DATASET_H
#define DATASET_H

#include "include/types.h"
#include "matrix.h"

typedef struct
{
    matrix_t **inputs;
    matrix_t **labels;
    u32 num_samples;
} dataset_t;

static dataset_t *dataset_create(u32 num_samples);
dataset_t *parse_file(char *path);
void dataset_destroy(dataset_t *data);
void dataset_print(const dataset_t *data);
void dataset_print_n(const dataset_t *data, u32 n);

#endif
