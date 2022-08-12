#include <stdio.h>
#include <omp.h>

#define CHUNK 100

void matrix_mult(double *A, double *B, double *C, int N){
	int i, j, k;
	register double sum;
     
    #pragma omp parallel for shared(A, B, C, N) private(i)
	for(i = 0; i < N; i++)
    #pragma omp parallel for schedule(dynamic, CHUNK)
		for(j = 0; j < N; j++){
			sum = 0;
			for(k = 0; k < N; k++)
				sum += A[i * N + k] * B[k * N + j];
			C[i * N + j] = sum;
		}
}
