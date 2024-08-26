#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <stdbool.h>

// #define DEBUG
#define REPS 15
#define DTYPE float
#define dim 3

int test();

typedef struct {
    DTYPE *allocated;
    DTYPE *aligned;
    intptr_t offset;
    intptr_t sizes[dim];
    intptr_t strides[dim];
} MemRefDescriptor;

void print_mat(const DTYPE *mat, const int r, const int c, const char *name);
DTYPE *init_mat(DTYPE *mat, const int b, const int r, const int c, const DTYPE val);
DTYPE *rand_mat(DTYPE *mat, const int b, const int r, const int c, const DTYPE min_val, const DTYPE max_val);
DTYPE *mat_mult(const int m, const int n, const int k,
                const DTYPE *a, const int lda,
                const DTYPE *b, const int ldb,
                      DTYPE *c, const int ldc);

extern void _mlir_ciface_matmul_mlir(MemRefDescriptor* arg1, MemRefDescriptor* arg2, MemRefDescriptor* arg3);
extern void _mlir_ciface_print(MemRefDescriptor* arg1);

MemRefDescriptor get_memref(DTYPE *mem, int b, int m, int n, int lda) {
    MemRefDescriptor descriptor;

    // Allocate continuous memory block for the matrix.
    descriptor.allocated = mem;
    if (descriptor.allocated == NULL) {
        // Handle allocation error. For simplicity, we exit in case of an error.
        // A real-world application should include proper error handling.
        exit(EXIT_FAILURE);
    }

    // No specific alignment is done here, so aligned points to the same location.
    descriptor.aligned = descriptor.allocated;

    // Assuming there isn't an additional offset in this case.
    descriptor.offset = 0;

    // Set sizes (dimensions) of the matrix.
    descriptor.sizes[0] = b; // Batch size
    descriptor.sizes[1] = m; // Number of rows
    descriptor.sizes[2] = n; // Number of columns

    // Calculate strides assuming row-major order.
    descriptor.strides[0] = m * lda; // Stride for batch dimension
    descriptor.strides[1] = lda;     // Stride for row dimension
    descriptor.strides[2] = 1;       // Stride for column dimension

    return descriptor;
}

// #define DEBUG
void print_mat(const DTYPE *mat, const int r, const int c, const char *name)
{
#ifdef DEBUG
    assert(mat && r>0 && c>0);
    printf("---- %s ----\n", name);
    for(int i=0; i<r; ++i)
    {
        for(int j=0; j<c; ++j)
        {
            printf("%.2f ", (float)mat[i*c+j]);
        }
        printf("\n");
    }
    printf("\n\n");
#endif
    return;
}


DTYPE *init_mat(DTYPE *mat, const int b, const int r, const int c, const DTYPE val)
{
    assert(mat && b>0 && r>0 && c>0);
    return rand_mat(mat, b, r, c, val, val);
}

DTYPE *zero_mat(size_t m, size_t n, size_t ldc, DTYPE *c) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            c[i*ldc + j] = 0.0;
        }
    }
    return c;
}

DTYPE *rand_mat(DTYPE *mat, const int b, const int r, const int c, const DTYPE min_val, const DTYPE max_val)
{
#ifdef DEBUG
    assert(mat && b > 0 && r > 0 && c > 0 && min_val <= max_val);
#endif
    static int rand_initialized = 0;
    if (!rand_initialized) {
        srand((unsigned int)time(NULL));
        rand_initialized = 1;
    }
    int range_val = ((int)(max_val - min_val) == 0) ? 0x7fffffff : (int)(max_val - min_val);
    for (int batch = 0; batch < b; ++batch)
    {
        for (int i = 0; i < r; ++i)
        {
            for (int j = 0; j < c; ++j)
            {
                mat[batch * r * c + i * c + j] = (rand() % range_val) + min_val;
            }
        }
    }
    return mat;
}

int compare(const void *a, const void *b) {
    double arg1 = *(const double *)a;
    double arg2 = *(const double *)b;
    if (arg1 < arg2) return -1;
    if (arg1 > arg2) return 1;
    return 0;
}

void print_statistics(double times[]) {
    // Calculate statistics
    // double min_time = times[0];
    // double max_time = times[0];
    // double sum_time = 0.0;
    // double sum_sq_time = 0.0;

    // for (int i = 0; i < REPS; ++i) {
    //     if (times[i] < min_time) min_time = times[i];
    //     if (times[i] > max_time) max_time = times[i];
    //     sum_time += times[i];
    //     // sum_sq_time += times[i] * times[i];
    // }

    // double avg_time = sum_time / REPS;
    // double variance = (sum_sq_time / REPS) - (avg_time * avg_time);
    // double stddev_time = sqrt(variance);

    // printf("Min time: %f second(s)\n", min_time);
    // printf("Max time: %f second(s)\n", max_time);
    // printf("Average time: %f second(s)\n", avg_time);
    // printf("Standard deviation: %f second(s)\n", stddev_time);


    qsort(times, REPS, sizeof(double), compare);
    // Calculate the median
    double median_time = times[REPS / 2];

    // Print the median
    printf("%f\n", median_time);
    // printf("%f", avg_time);
}

int main(int argc, char *argv[])
{
    // init
    // const int loop_times = 10;

    double start_time = 0.0;
    double end_time   = 0.0;

    const int B = 6;  // Batch size
    const int m = 196;
    const int n = 256;
    const int k = 2304;
    const int lda = k;
    const int ldb = n;
    const int ldc = n;

    DTYPE *a = calloc(B * m * lda, sizeof(DTYPE));
    DTYPE *b = calloc(B * k * ldb, sizeof(DTYPE));
    DTYPE *c = calloc(B * m * ldc, sizeof(DTYPE));
    // print_mat(a, m, lda, "a");
    // print_mat(b, k, ldb, "b");
    // print_mat(c, m, ldc, "c");

    a = rand_mat(a, B, m, k, 1.0, 10.0);
    b = rand_mat(b, B, k, n, 1.0, 10.0);
    c = init_mat(c, B, m, n, 0.0);

    // printf("---------------------------------------\n");
    // // matmul mlir
    MemRefDescriptor memref_a = get_memref(a, B, m, n, lda);
    MemRefDescriptor memref_b = get_memref(b, B, m, n, lda);
    MemRefDescriptor memref_c = get_memref(c, B, m, n, lda);
    double mlir_times[REPS];

    for(int idx=0; idx<REPS; ++idx) {
        // zero_mat(b, m, n, ldc, memref_c_.aligned);
        init_mat(memref_c.aligned, B, m, ldc, 0);
        start_time = omp_get_wtime();
        _mlir_ciface_matmul_mlir(&memref_c, &memref_a, &memref_b);
        end_time = omp_get_wtime();
        // printf("mlir: idx:%d time:%f second(s)\n", idx, end_time-start_time);
        mlir_times[idx] = end_time - start_time;

        // Dont perform all repetitions if the time is too high
        if (mlir_times[idx] > 1.0) {
            // printf("Time too high, aborting this configuration\n");
            for (int i = idx; i < REPS; ++i) {
                mlir_times[i] = end_time - start_time;
            }
            break;
        }
    }
    print_statistics(mlir_times);

    // MemRefDescriptor mlir_memref = get_memref(c, m, n, ldc);
    // // _mlir_ciface_print(&memref_c);
    // print_mat(memref_c.allocated, m, ldc, "mlir c");
    // c = init_mat(c, m, ldc, 0);


    // print_mat(a, m, lda, "init a");
    // print_mat(b, k, ldb, "init b");
    // print_mat(c, m, ldc, "init c");
    // print_mat(c, m, ldc, "mat mul c");

    if(a) free(a); a = NULL;
    if(b) free(b); b = NULL;
    if(c) free(c); c = NULL;

    // test();

    return 0;
}
