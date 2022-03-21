#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utils.h"
#ifdef _OPENMP
    #include <omp.h>
#endif

long double calculate_resid(double *u, int* f, double N) {
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

void gauss_seidel(int N) {
    
    double* u = (double*) malloc((N+2)*(N+2)*sizeof(double));
    for (long i = 0; i < (N+2)*(N+2); i++) u[i] = 0;
    double n = (double) N;
    int elmts = N+2;
    double h = 1/(n+1);
    int* f = (int*) malloc((N+2)*(N+2)*sizeof(int));
    for (long i = 0; i < (N+2)*(N+2); i++) f[i] = 1;
    double residual = calculate_resid(u,f,n);
    printf("%fl",residual);
    double resid_temp=0;

    for (long k = 0; k < 5000; k++) {
        #ifdef _OPENMP
            #pragma omp parallel //num_threads(4)
        #endif

        {
            //red points
            #ifdef _OPENMP
                #pragma omp for collapse(2)
            #endif
            for (long j=1; j < N+1; j+=2) {
                for (long i = 1; i < N+1; i+=2) {
                    u[i*elmts+j]=(u[(i-1)*elmts+j]+u[i*elmts+j-1]+u[(i+1)*elmts+j]+u[i*elmts+j+1]+f[i*elmts+j]*h*h)/4.0;
                }
            }
            #ifdef _OPENMP
                #pragma omp for collapse(2)
            #endif
            for (long j=2; j < N+1; j+=2) {
                for (long i = 2; i < N+1; i+=2) {
                    u[i*elmts+j]=(u[(i-1)*elmts+j]+u[i*elmts+j-1]+u[(i+1)*elmts+j]+u[i*elmts+j+1]+f[i*elmts+j]*h*h)/4.0;
                }
            }
            
            //black points
            #ifdef _OPENMP
                #pragma omp barrier
                #pragma omp for collapse(2)
            #endif
            for (long j=1;j < N+1; j+=2) {
                for (long i = 2; i < N+1; i+=2) {
                    u[i*elmts+j]=(u[(i-1)*elmts+j]+u[i*elmts+j-1]+u[(i+1)*elmts+j]+u[i*elmts+j+1]+f[i*elmts+j]*h*h)/4.0;
                }
            }

            #ifdef _OPENMP
                #pragma omp for collapse(2)
            #endif
            for (long j=2;j < N+1; j+=2) {
                for (long i = 1; i < N+1; i+=2) {
                    u[i*elmts+j]=(u[(i-1)*elmts+j]+u[i*elmts+j-1]+u[(i+1)*elmts+j]+u[i*elmts+j+1]+f[i*elmts+j]*h*h)/4.0;
                }
            }
        }   
        resid_temp = calculate_resid(u,f,n);
        if ((residual / resid_temp) >= 1e6) {
            printf("converged!!!\n");
            return;
            }   
        printf("iteration: %ld residual: %f \n", k, resid_temp);
        
    }
    free(u);
    free(f);
}

int main(int argc, char** argv) {
    int n = read_option<int>("-n", argc, argv,"200");
    Timer t;
    t.tic();
    gauss_seidel(n);
    printf("matrix N = %d, Time: %10f\n", n, t.toc());
    return 0;
}
