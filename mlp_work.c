#include <math.h>
#include <stdio.h>
#include <stdlib.h>
int sizes[] = {2, 3, 4, 3, 1};
int layers = sizeof(sizes) / sizeof(int);
int batch_size = 1;

struct mlp {
    double** biases;
    double*** weights;
};

struct bp {
    double **z, **a, **delta;
};

double random_(double low, double high) { return (((high - low) * (double)rand() / RAND_MAX) + low); }
double sigmoid(double x) { return 1 / (1 + exp(-x)); }
double d_sigmoid(double x) { return sigmoid(x) * (1 - sigmoid(x)); }

double** alloc_biases() {
    double** arr = malloc((layers - 1) * sizeof(double*));
    for (int l = 0; l < layers - 1; ++l) {
        arr[l] = malloc(sizes[l + 1] * sizeof(double));
        for (int i = 0; i < sizes[l + 1]; ++i) {
            arr[l][i] = 0;
        }
    }
    return arr;
}

double*** alloc_weights() {
    double*** weights = malloc(layers * sizeof(double**));
    for (int l = 0; l < layers - 1; ++l) {
        weights[l] = malloc(sizes[l + 1] * sizeof(double*));
        for (int i = 0; i < sizes[l + 1]; ++i) {
            weights[l][i] = malloc(sizes[l] * sizeof(double));
            for (int j = 0; j < sizes[l]; ++j) {
                weights[l][i][j] = random_(-1, 1);
            }
        }
    }

    return weights;
}

double** alloc_cache() {
    double** arr = malloc(layers * sizeof(double*));
    for (int l = 0; l < layers; ++l) {
        arr[l] = malloc(sizes[l] * sizeof(double));
        for (int i = 0; i < sizes[l]; ++i) {
            arr[l][i] = 0;
        }
    }
    return arr;
}

void forward(double* input, double* output, struct mlp* net, struct bp* cache) {
    for (int i = 0; i < sizes[0]; ++i) {
        cache->a[0][i] = input[i];
    }

    for (int l = 1; l < layers; ++l) {
        for (int i = 0; i < sizes[l]; ++i) {
            cache->z[l][i] = net->biases[l - 1][i];
            for (int j = 0; j < sizes[l - 1]; ++j) {
                cache->z[l][i] +=
                    net->weights[l - 1][i][j] * cache->a[l - 1][j];
            }
            cache->a[l][i] = sigmoid(cache->z[l][i]);
        }
    }

    for (int i = 0; i < sizes[layers - 1]; ++i) {
        output[i] = cache->z[layers - 1][i];
    }
}

double loss(double* output, double* ground_truth) {
    double sum = 0;

    for (int i = 0; i < sizes[layers - 1]; ++i) {
        sum += (output[i] - ground_truth[i]) * (output[i] - ground_truth[i]);
    }

    return .5 * sum;
}

void backward(double* ground_truth, struct mlp* net, struct bp* cache) {
    for (int i = 0; i < sizes[layers - 1]; ++i) {
        cache->delta[layers - 1][i] =
            -(ground_truth[i] - cache->z[layers - 1][i]);
    }

    for (int l = layers - 2; l >= 0; --l) {
        for (int i = 0; i < sizes[l]; ++i) {
            double sum = 0;
            for (int j = 0; j < sizes[l + 1]; ++j) {
                sum +=
                    cache->delta[l + 1][j] * net->weights[l][j][i];
            }
            cache->delta[l][i] = sum * d_sigmoid(cache->z[l][i]);
        }
    }
}

void optimize(double lr, struct mlp* net, struct bp* cache) {
    for (int l = 0; l < layers - 1; ++l) {
        for (int i = 0; i < sizes[l + 1]; ++i) {
            for (int j = 0; j < sizes[l]; ++j) {
                net->weights[l][i][j] -= lr * (cache->delta[l + 1][i] * cache->a[l][j]);
            }
            net->biases[l][i] -= lr * cache->delta[l + 1][i]; 
        }
    }
}

void zero_cache(double** arr) {
    for (int l = 0; l < layers; ++l) {
        for (int i = 0; i < sizes[l]; ++i) {
            arr[l][i] = 0;
        }
    }
}

void zero_grad(struct bp* cache) {
    zero_cache(cache->z);
    zero_cache(cache->delta);
    zero_cache(cache->a);
}


void print_vector(double* vec, int _dim) {
    int first = 1;
    printf("(");
    for (int dim = 0; dim < _dim; ++dim) {
        if (!first) printf(", ");
        first = 0;
        printf("%.3f", vec[dim]);
    }
    printf(")");
}

int main() {
    struct mlp net;
    struct bp cache;

    double input[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    double ground_truth[4][1] = {{0}, {1}, {1}, {0}};
    double output[1];

    cache.z = alloc_cache();
    cache.a = alloc_cache();
    cache.delta = alloc_cache();

    net.biases = alloc_biases();
    net.weights = alloc_weights();

    for (int epoch = 0; epoch < 50000; ++epoch) {
        for (int n = 0; n < 4; ++n) {
            forward(input[n], output, &net, &cache);
            if (epoch % 1000 == 0)
                printf("epoch %d, sample %d, loss %.6f\n", epoch, n, loss(output, ground_truth[n]));
            backward(ground_truth[n], &net, &cache);
            optimize(1, &net, &cache);
            zero_grad(&cache);
        }
    }

    for (int i = 0; i < 4; ++i) {
        forward(input[i], output, &net, &cache);
        printf("input: ");
        print_vector(input[i], sizes[0]);

        printf(" output: ");
        print_vector(output, sizes[layers - 1]);

        printf(" ground_truth: ");
        print_vector(ground_truth[i], sizes[layers - 1]);
        puts("");
    }
}