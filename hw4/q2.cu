#include <algorithm>
#include <stdio.h>
#include "utils.h"
#include <stdlib.h>
#include <math.h>
#ifdef _OPENMP
    #include <omp.h>
#endif

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32

long double calculate_resid(double *u, long* f, double N) {
    double resid = 0.0;
    double h = 1/(N+1); 
    double err = 0.0;
    int int_n = (int) N;
    int elmts = int_n + 2;
    for (long i = 1; i <= N; i++) { 
        for (long j = 1; j <= N; j++) {
            err = fabs(f[i*elmts+j]-(4*u[i*elmts+j]-u[(i-1)*elmts+j]-u[i*elmts+j-1]-u[(i+1)*elmts+j]-u[i*elmts+j+1])/h/h);
            resid = std::max(resid,err);
        } 
    }
    return resid;
}

void jacobi_cpu(double* u, double* temp_u, long* f, int N, double h) {
    int elmts = N+2;
    double residual = calculate_resid(u,f,N);

    for (long k = 0; k < 5000; k++) {
        #ifdef _OPENMP
            #pragma omp parallel
        #endif
        {
            #ifdef _OPENMP
             #pragma omp for
            #endif
            for (long i = 0; i < (N+2)*(N+2); i++) {
                temp_u[i] = u[i];
            } 
            #ifdef _OPENMP
                #pragma omp barrier
                #pragma omp for collapse(2)
            #endif  
            for (long j=1; j < N+1; j+=1) {
                    for (long i = 1; i < N+1; i+=1) {
                        u[i*elmts+j]=(temp_u[(i-1)*elmts+j]+temp_u[i*elmts+j-1]+temp_u[(i+1)*elmts+j]+temp_u[i*elmts+j+1]+f[i*elmts+j]*h*h)/4.0;
                    }
                }
        }
        double resid_temp = calculate_resid(u,f,N);
        if ((residual / resid_temp) >= 1e6) {
            printf("converged!!!, takes %ld\n",k);
            return;
            }   
        //printf("iteration: %ld residual: %f \n", k, resid_temp);
        
    }
}

__global__
void jacobi_gpu(double* u, double* u_old, long* f, const int N, const double h){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i>0 && i<N-1 && j >0 && j<N-1) {
        u[i*N+j] = (u_old[(i-1)*N+j] + u_old[(i+1)*N+j] + u_old[i*N+j+1] + u_old[i*N+j-1] +f[i*N+j]*h*h)/4.0;
    }
}

int main(int argc, char** argv) {
    int N = read_option<int>("-n", argc, argv,"254");
    double* u = (double*) malloc((N+2)*(N+2)*sizeof(double));
    double* u_old = (double*) malloc((N+2)*(N+2)*sizeof(double));
    double* u_gpu = (double*) malloc((N+2)*(N+2)*sizeof(double));
    long* f = (long*) malloc((N+2)*(N+2)*sizeof(long));
    double n = (double) N;
    double h = 1/(n+1);
   

    // initialization
    for (long i = 0; i < (N+2)*(N+2); i++) u[i] = 0;
    for (long i = 0; i < (N+2)*(N+2); i++) f[i] = 1.0;
    double residual = calculate_resid(u,f,N);


    //cuda malloc
    double *u_d, *u_d_old;
    long* f_d;
    cudaMalloc(&u_d, (N+2)*(N+2)*sizeof(double));
    cudaMalloc(&u_d_old, (N+2)*(N+2)*sizeof(double));
    cudaMalloc(&f_d, (N+2)*(N+2)*sizeof(long));

    double tt = omp_get_wtime();
    cudaMemcpy(u_d, u, (N+2)*(N+2)*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(f_d, f, (N+2)*(N+2)*sizeof(long), cudaMemcpyHostToDevice);
    double copy_t = omp_get_wtime()-tt;

     //cpu version 
    tt = omp_get_wtime();
    jacobi_cpu(u,u_old,f,N,h);
    printf("CPU %f s\n", omp_get_wtime()-tt);

    // gpu version 
    dim3 block(BLOCK_SIZE_X,BLOCK_SIZE_Y);
    dim3 grid((N+2)/BLOCK_SIZE_X, (N+2)/BLOCK_SIZE_Y);

    tt = omp_get_wtime();
    for (int k=0; k<5000; k+=2) {
        jacobi_gpu<<<grid,block>>>(u_d_old,u_d,f_d,N+2,h);
        cudaDeviceSynchronize();
        jacobi_gpu<<<grid,block>>>(u_d,u_d_old, f_d,N+2,h);
        cudaDeviceSynchronize();
        // cudaMemcpy(u_gpu, u_d, (N+2)*(N+2)*sizeof(double), cudaMemcpyDeviceToHost);
        // double resid_gpu = calculate_resid(u_gpu,f,N);
        // if ((residual / resid_gpu) >= 1e6) {
        //     printf("converged!!!\n");
        //     printf("GPU %f s, %f s, takes %ld, resid %f \n", omp_get_wtime()-tt+copy_t, omp_get_wtime()-tt, k+1, resid_gpu);
        //     return 0;
        //     }   
        // printf("iteration: %ld residual: %f \n", k, resid_gpu);
    }
    cudaMemcpy(u_gpu, u_d, (N+2)*(N+2)*sizeof(double), cudaMemcpyDeviceToHost);
    tt = omp_get_wtime()-tt;
    printf("GPU %f s, %f s\n", tt+copy_t, tt);

    double error = 0;
    for (int i=0; i<N; i++) error += (u_gpu[i] - u[i]);

    printf("Error: %f \n", error);
   
    free(u);
    free(u_old);
    free(u_gpu);
    free(f);
    cudaFree(u_d);
    cudaFree(f_d);
    cudaFree(u_d_old);
    return 0;

}