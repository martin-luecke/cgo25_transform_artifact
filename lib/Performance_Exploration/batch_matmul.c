#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <stdbool.h>

// #define DEBUG
#define REPS 10
#define DTYPE float
#define dim 3


#define Batch 6
#define M 196
#define N 256
#define K 2304

typedef struct {
    DTYPE *allocated;
    DTYPE *aligned;
    intptr_t offset;
    intptr_t sizes[dim];
    intptr_t strides[dim];
} MemRefDescriptor;

void print_mat(const DTYPE *mat, const int r, const int c, const char *name);
void init_mat(DTYPE *mat, const int b, const int r, const int c, const DTYPE val);
void rand_mat(DTYPE *mat, const int b, const int r, const int c, const DTYPE min_val, const DTYPE max_val);
extern void _mlir_ciface_matmul_mlir(MemRefDescriptor* arg1, MemRefDescriptor* arg2, MemRefDescriptor* arg3);
extern void _mlir_ciface_print(MemRefDescriptor* arg1);


MemRefDescriptor get_memref(DTYPE *mem, int b, int m, int n) {
    MemRefDescriptor descriptor;
    descriptor.allocated = mem;
    if (descriptor.allocated == NULL) {
        // Handle allocation error.
        exit(EXIT_FAILURE);
    }

    // No specific alignment is done here, so aligned points to the same location.
    descriptor.aligned = descriptor.allocated;

    // No additional offset in this case.
    descriptor.offset = 0;

    // Set sizes of the matrix.
    descriptor.sizes[0] = b; // Batch size
    descriptor.sizes[1] = m; // Number of rows
    descriptor.sizes[2] = n; // Number of columns

    // Calculate strides following row-major order.
    descriptor.strides[0] = m * n; // Stride for batch dimension
    descriptor.strides[1] = n;     // Stride for row dimension
    descriptor.strides[2] = 1;     // Stride for column dimension

    return descriptor;
}

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


void init_mat(DTYPE *mat, const int b, const int r, const int c, const DTYPE val)
{
    assert(mat && b>0 && r>0 && c>0);
    rand_mat(mat, b, r, c, val, val);
}

void zero_mat(size_t b, size_t m, size_t n, size_t ldc, DTYPE (*C)[M][N]) {
    for (size_t batch = 0; batch < b; ++batch) {
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                C[batch][i][j] = 0.0;
            }
        }
    }
}

void rand_mat(DTYPE *mat, const int b, const int r, const int c, const DTYPE min_val, const DTYPE max_val)
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
}


void batch_mat_mul_omp(const DTYPE (*A)[M][K], const int lda,
                     const DTYPE (*B)[K][N], const int ldb,
                     DTYPE (*C)[M][N], const int ldc)
{
#ifdef DEBUG
    assert(a && b && c && lda > 0 && ldb > 0 && ldc > 0);
    assert(B > 0 && M > 0 && N > 0 && K > 0);
#endif

    int b, i, j, k;
    for (b = 0; b < Batch; b++) {
        #pragma omp tile sizes(32, 32)
        for (i = 0; i < M; i++) {
            for (j = 0; j < N; j++) {
                for (k = 0; k < K; k++) {
                    C[b][i][j] += A[b][i][k] * B[b][k][j];
                }
            }
        }
    }
}

void batch_mat_mul(const DTYPE (*A)[M][K], const int lda,
                     const DTYPE (*B)[K][N], const int ldb,
                     DTYPE (*C)[M][N], const int ldc)
{
#ifdef DEBUG
    assert(a && b && c && lda > 0 && ldb > 0 && ldc > 0);
    assert(B > 0 && M > 0 && N > 0 && K > 0);
#endif

    int b, i, j, k;
    for (b = 0; b < Batch; b++) {
        for (i = 0; i < M; i++) {
            for (j = 0; j < N; j++) {
                for (k = 0; k < K; k++) {
                    C[b][i][j] += A[b][i][k] * B[b][k][j];
                }
            }
        }
    }
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
    double min_time = times[0];
    double max_time = times[0];
    double sum_time = 0.0;
    double sum_sq_time = 0.0;

    qsort(times, REPS, sizeof(double), compare);
    // Calculate the median
    double median_time = times[REPS / 2];

    // Print the median
    for (int i = 0; i < REPS; ++i) {
        if (times[i] < min_time) min_time = times[i];
        if (times[i] > max_time) max_time = times[i];
        sum_time += times[i];
        sum_sq_time += times[i] * times[i];
    }

    double avg_time = sum_time / REPS;
    double variance = (sum_sq_time / REPS) - (avg_time * avg_time);
    // double stddev_time = sqrt(variance);

    printf("Min time: %f second(s)\n", min_time);
    printf("Max time: %f second(s)\n", max_time);
    printf("Average time: %f second(s)\n", avg_time);
    printf("Median time: %f second(s)\n", median_time);
    // printf("Standard deviation: %f second(s)\n", stddev_time);
}

int check_match(DTYPE (*gold)[M][N], DTYPE (*computed)[M][N]) {
    for (int b = 0; b < Batch; b++) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                if (gold[b][i][j] != computed[b][i][j]) {
                    printf("mismatch at %i %i %i: %d vs %d \n", b, i, j, gold[b][i][j], computed[b][i][j]);
                    return 0; // Return 0 if any element does not match
                }
            }
        }
    }
    return 1; // Return 1 if all elements match
}


int main(int argc, char *argv[])
{
    // init
    const int loop_times = 10;

    double start_time = 0.0;
    double end_time   = 0.0;

    const int lda = K;
    const int ldb = N;
    const int ldc = N;

    DTYPE (*A)[M][K] = calloc(Batch, sizeof(DTYPE[M][K]));
    DTYPE (*B)[K][N] = calloc(Batch, sizeof(DTYPE[K][N]));
    DTYPE (*C)[M][N] = calloc(Batch, sizeof(DTYPE[M][N]));
    DTYPE (*gold)[M][N] = calloc(Batch, sizeof(DTYPE[M][N]));

    rand_mat(A, Batch, M, K, 0.1, 1.0);
    rand_mat(B, Batch, K, N, 0.1, 1.0);
    init_mat(C, Batch, M, N, 0.0);
    init_mat(gold, Batch, M, N, 0.0);

    for (int b=0; b<Batch; b++) {
        for (int i = 0; i < M; i++) {
            for(int j = 0; j < N; j++) {
                for(int k = 0; k < K; k++) {
                    gold[b][i][j] += A[b][i][k] * B[b][k][j];
                }
            }
        }
    }

    // mat mul reference
    bool do_reference = false;
    if (do_reference) {
        double reference_times[REPS];
        for(int idx=0; idx<REPS; ++idx) {
            init_mat(C, Batch, M, N, 0.0);
            start_time = omp_get_wtime();
            batch_mat_mul(A, lda, B, ldb, C, ldc);
            end_time = omp_get_wtime();
            reference_times[idx] = end_time - start_time;
        }
        printf("reference times:\n");
        print_statistics(reference_times);
        printf("matches gold: %i\n", check_match(gold, C));
        printf("---------------------------------------\n");
    }

    //////////////////////////////////////////////////////////////////////

    // matmul mlir
    MemRefDescriptor memref_a = get_memref(A, Batch, M, K);
    MemRefDescriptor memref_b = get_memref(B, Batch, K, N);
    MemRefDescriptor memref_c = get_memref(C, Batch, M, N);
    double mlir_times[REPS];

    for(int idx=0; idx<REPS; ++idx) {
        init_mat(memref_c.aligned, Batch, M, N, 0.0);

        start_time = omp_get_wtime();
        _mlir_ciface_matmul_mlir(&memref_c, &memref_a, &memref_b);
        end_time = omp_get_wtime();
        mlir_times[idx] = end_time - start_time;
    }
    printf("Transform runtimes:\n");
    print_statistics(mlir_times);
    printf("matches gold: %i\n", check_match(gold, memref_c.aligned));

    printf("---------------------------------------\n");

    // OpenMP
    double omp_times[REPS];
    for(int idx=0; idx<REPS; ++idx) {
        init_mat(C, Batch, M, N, 0.0);

        start_time = omp_get_wtime();
        batch_mat_mul_omp(A, lda, B, ldb, C, ldc);
        end_time = omp_get_wtime();
        omp_times[idx] = end_time - start_time;
    }
    printf("OpenMP runtimes:\n");
    print_statistics(omp_times);
    printf("matches gold: %i\n", check_match(gold, C));


    free(A);
    free(B);
    free(C);
    free(gold);

    return 0;
}