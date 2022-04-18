#include <stdio.h>
#include <math.h>
#include <omp.h>

#define N 4096 // row size 
#define M 8192 // column size
#define BLOCK_SIZE 1024

void Check_CUDA_Error(const char *message){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
    exit(-1);
  }
}

void matrix_vec (double* A, double* V, double *C){
    //#pragma omp parallel for collapse(2)
    for (long i = 0; i < N; i++) {
        double inner_prod = 0;
        for (long j = 0; j < M; j++) {
            //#pragma omp atomic update 
            inner_prod += A[i*M+j] * V[j];
        }
        C[i] = inner_prod;
    }
}

__global__ 
void matrix_vec_kernel(double* A, double* V, double* C, double* temp) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        double value = 0;
        for (long i = 0; i < M; i++) {
            value += A[idx*M+i] * V[i];
        }
        C[idx] = value;
    }
}

int main(int argc, char** argcv) {
    long col_size = M;
    long row_size = N;

    double* A = (double*) malloc(col_size*row_size * sizeof(double));
    double* V = (double*) malloc(col_size * sizeof(double));
    double* C = (double*) malloc(row_size * sizeof(double));
    double* C_ref = (double*) malloc(row_size * sizeof(double));

    //initialization
    for (long i = 0; i < col_size*row_size; i++) A[i] = drand48();
    for (long i = 0; i < col_size; i++) V[i] = drand48();
    for (long i = 0; i < row_size; i++) C[i] = 0.0;
    for (long i = 0; i < row_size; i++) C_ref[i] = 0.0;

    //cpu
    double tt = omp_get_wtime();
    matrix_vec(A,V,C_ref);
    printf("CPU %f s\n", omp_get_wtime()-tt);
    double cpu_bandwidth = 3*M*N*sizeof(double)/(omp_get_wtime()-tt)/1e9;
    printf("CPU bandwidth = %f GB/s\n", cpu_bandwidth);

    double *A_d, *V_d, *C_d, *temp;
    cudaMalloc(&A_d, col_size*row_size*sizeof(double));
    Check_CUDA_Error("malloc x failed");
    cudaMalloc(&V_d, col_size*sizeof(double));
    cudaMalloc(&C_d, row_size*sizeof(double));
    cudaMalloc(&temp, col_size*row_size*sizeof(double));

    tt = omp_get_wtime();
    cudaMemcpy(A_d, A, col_size*row_size*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(V_d, V, col_size*sizeof(double), cudaMemcpyHostToDevice);
    double ttinner = omp_get_wtime();
    matrix_vec_kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(A_d,V_d,C_d,temp);
    cudaDeviceSynchronize();
    ttinner = omp_get_wtime() - ttinner;
    cudaMemcpy(C, C_d, row_size*sizeof(double), cudaMemcpyDeviceToHost);
    printf("GPU %f s, %f s\n", omp_get_wtime()-tt, ttinner);

    double bandwidth = 3*M*N*sizeof(double)/(omp_get_wtime()-tt)/1e9;

    printf("GPU bandwidth = %f GB/s\n", bandwidth);

    // for (long i = 0; i < row_size; i++) printf("%f\n",C[i]);
    // for (long i = 0; i < row_size; i++) printf("c_ref: %f\n",C_ref[i]);

    double err = 0;
    for (long i = 0; i < N; i++) err += fabs(C[i]-C_ref[i]);
    printf("Error = %f\n", err);

    cudaFree(A_d);
    cudaFree(V_d);
    cudaFree(C_d);
    cudaFree(temp);

    free(A);
    free(V);
    free(C);
    free(C_ref);

    return 0;
}