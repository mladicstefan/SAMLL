#pragma once

#include "include/types.h"
#include <math.h>

f64 reLU(f64 z);
f64 sigmoid(f64 z);

f64 reLU(f64 z) { return (z <= 0) ? 0 : z; }

f64 reLU_derivative(f64 x) { return x > 0 ? 1.0 : 0.0; }

f64 sigmoid(f64 z) { return 1 / (1 + exp(-z)); }

f64 sigmoid_derivative_from_output(f64 sigmoid_output)
{
    return sigmoid_output * (1.0 - sigmoid_output);
}
