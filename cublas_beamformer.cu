//#include <stdio.h>
//#include <stdlib.h>
//#include <cstdlib>
//#include <curand.h>
//#include <assert.h>
//#include <unistd.h>
//#include <cublas_v2.h>
//#include <iostream>
//#include <complex.h>
//#include <math.h>
//#include <cuComplex.h>
//#include <cuda_runtime.h>
//#include "cublas_beamformer.h"
//
//using namespace std;
//
//// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on GPU
//void GPU_fill(cuComplex *A, int nr_rows_A, int nr_cols_A) {
//
//	/*
//	// Create a pseudo-random number generator
//	curandGenerator_t prng;
//	curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
//
//	// Set the seed for the random number generator using the system clock
//	curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());
//
//	// Fill the array with random numbers on the device
//	curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);
//	*/
////	float *G;
//
//	cuComplex *G;
//	G = new cuComplex[nr_rows_A*nr_cols_A];
//	for(int i = 0; i < nr_rows_A*nr_cols_A; i++){
//		G[i].x = 1;
//		G[i].y = 0;
//	}
//	cudaMemcpy(A,G,nr_rows_A * nr_cols_A * sizeof(cuComplex),cudaMemcpyHostToDevice);
//	delete[] G;
//}
//
//
////Print matrix A(nr_rows_A, nr_cols_A) storage in column-major format
//void print_matrix(const cuComplex *A, int nr_rows_A, int nr_cols_A) {
//	for(int i = 0; i < nr_rows_A; ++i){
//		for(int j = 0; j < nr_cols_A; ++j){
//			std::cout << A[j * nr_rows_A + i].x + A[j * nr_rows_A + i].y << " ";
//		}
//		std::cout << std::endl;
//	}
//	std::cout << std::endl;
//}
//
////void beamform(const cuComplex * h_A,
////	const cuComplex * h_B,
////	cuComplex * h_C) { {
//int main(){
//	// Allocate 3 arrays on CPU
//	cudaError_t cudaStat;
//
//	int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;
//
// 	nr_rows_A = N_TIME_STI;
// 	nr_cols_A = N_ELE;
// 	nr_rows_B = N_ELE;
// 	nr_cols_B = N_BEAM;
// 	nr_rows_C = N_TIME_STI;
// 	nr_cols_C = N_BEAM;
//
//	// for simplicity we are going to use square arrays
//	//nr_rows_A = nr_cols_A = nr_rows_B = nr_cols_B = nr_rows_C = nr_cols_C = 3;
//
// 	cuComplex *h_A = (cuComplex *)malloc(nr_rows_A * nr_cols_A * N_STI * N_BIN * sizeof(cuComplex));
// 	cuComplex *h_B = (cuComplex *)malloc(nr_rows_B * nr_cols_B * N_STI * N_BIN * sizeof(cuComplex));
// 	cuComplex *h_C = (cuComplex *)malloc(nr_rows_C * nr_cols_C * N_STI * N_BIN * sizeof(cuComplex));
//
//	// Allocate 3 arrays on GPU
// 	cuComplex *d_A, *d_B, *d_C;
//	cudaMalloc(&d_A,nr_rows_A * nr_cols_A * N_STI * N_BIN * sizeof(cuComplex));
//	cudaMalloc(&d_B,nr_rows_B * nr_cols_B * N_STI * N_BIN * sizeof(cuComplex));
//	cudaMalloc(&d_C,nr_rows_C * nr_cols_C * N_STI * N_BIN * sizeof(cuComplex));
//
//	// Fill the arrays A and B on GPU with random numbers
//	GPU_fill(d_A, nr_rows_A*N_STI, nr_cols_A*N_BIN);
//	GPU_fill(d_B, nr_rows_B*N_STI, nr_cols_B*N_BIN);
//
//	// Optionally we can copy the data back on CPU and print the arrays
//	cudaMemcpy(h_A,d_A,nr_rows_A * nr_cols_A * N_STI * N_BIN * sizeof(cuComplex),cudaMemcpyDeviceToHost);
//	cudaMemcpy(h_B,d_B,nr_rows_B * nr_cols_B * N_STI * N_BIN * sizeof(cuComplex),cudaMemcpyDeviceToHost);
////	std::cout << "A =" << std::endl;
////	print_matrix(h_A, nr_rows_A*N_STI, nr_cols_A*N_BIN);
////	std::cout << "B =" << std::endl;
////	print_matrix(h_B, nr_rows_B*N_STI, nr_cols_B*N_BIN);
//
//	cuComplex **h_arr_A = 0; cuComplex **h_arr_B = 0; cuComplex **h_arr_C = 0;
//	//New code ////////
//	h_arr_A = (cuComplex **)malloc(nr_rows_A * nr_cols_A *N_STI*N_BIN*sizeof(cuComplex*));
//	h_arr_B = (cuComplex **)malloc(nr_rows_B * nr_cols_B *N_STI*N_BIN*sizeof(cuComplex*));
//	h_arr_C = (cuComplex **)malloc(nr_rows_C * nr_cols_C *N_STI*N_BIN*sizeof(cuComplex*));
//
//	for(int i = 0; i < N_STI*N_BIN; i++){
//		h_arr_A[i] = d_A + i*N_ELE*N_TIME_STI;
//		h_arr_B[i] = d_B + i*N_ELE*N_BEAM;
//		h_arr_C[i] = d_C + i*N_TIME_STI*N_BEAM;
//	}
//
////	float At[40*38];
////	cudaMemcpy(At,h_arr_B[1],38*7*sizeof(float),cudaMemcpyDeviceToHost);
////	std::cout << "At =" << std::endl;
////	for(int i =0; i<38*7;i++)
////		std::cout << i << ":\t" << At[i] << std::endl;
//
//	cuComplex **d_arr_A = 0; cuComplex **d_arr_B = 0; cuComplex **d_arr_C = 0;
//	cudaStat = cudaMalloc(&d_arr_A,nr_rows_A * nr_cols_A * N_STI * N_BIN * sizeof(cuComplex*));
//	assert(!cudaStat);
//	cudaStat = cudaMalloc(&d_arr_B,nr_rows_B * nr_cols_B * N_STI * N_BIN * sizeof(cuComplex*));
//	assert(!cudaStat);
//	cudaStat = cudaMalloc(&d_arr_C,nr_rows_C * nr_cols_C * N_STI * N_BIN * sizeof(cuComplex*));
//	assert(!cudaStat);
//
//	//cudaMemcpy(d_arr_A,h_arr_A[0],nr_rows_A * nr_cols_A * N_STI * N_BIN * sizeof(float*),cudaMemcpyHostToDevice);
//
//	//printf("H_arr_A %d",&h_arr_A[0]);
//
//	cudaStat = cudaMemcpy(d_arr_A,h_arr_A,nr_rows_A * nr_cols_A * N_STI * N_BIN * sizeof(cuComplex*),cudaMemcpyHostToDevice);
//	assert(!cudaStat);
//	cudaStat = cudaMemcpy(d_arr_B,h_arr_B,nr_rows_B * nr_cols_B * N_STI * N_BIN * sizeof(cuComplex*),cudaMemcpyHostToDevice);
//	assert(!cudaStat);
//	cudaStat = cudaMemcpy(d_arr_C,h_arr_C,nr_rows_C * nr_cols_C * N_STI * N_BIN * sizeof(cuComplex*),cudaMemcpyHostToDevice);
//	assert(!cudaStat);
//	// Multiply A and B on GPU
//
//	//gpu_blas_mmul(d_arr_A, d_arr_B, d_arr_C, nr_rows_A, nr_rows_B, nr_cols_A);
//	int lda=nr_rows_A,ldb=nr_rows_B,ldc=nr_rows_A;
//	cuComplex alf;
//	cuComplex bet;
//
//	alf.x = 1;
//	alf.y = 0;
//	bet.x = 0;
//	bet.y = 0;
////	const float *alpha = &alf;
////	const float *beta = &bet;
//	//New variables
//	int batchCount = N_STI*N_BIN;
//
//	// Create a handle for CUBLAS
//	cublasHandle_t handle;
//	cublasCreate(&handle);
//
//	cublasStatus_t stat;
//	// Do the actual multiplication
////	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
//
//	stat = cublasCgemmBatched(
//			handle,
//			CUBLAS_OP_N,
//			CUBLAS_OP_N,
//			nr_rows_A,
//			nr_cols_A,
//			nr_rows_B,
//			&alf,
//			(const cuComplex **)d_arr_A,
//			lda,
//			(const cuComplex **)d_arr_B,
//			ldb,
//			&bet,
//			(cuComplex **)d_arr_C,
//			ldc,
//			batchCount);
//
//
//	if(stat != CUBLAS_STATUS_SUCCESS){
//		cerr << "cublasSgemmBatched failed" << endl;
//		exit(1);
//	}
//	assert(!cudaGetLastError());
//
//	//////////////////////
//
//
//	// Multiply A and B on GPU
//
//	//gpu_blas_mmul(d_A, d_B, d_C, nr_rows_A, nr_rows_B, nr_cols_A);
//
//	// Copy (and print) the result on host memory
//	cudaMemcpy(h_C,h_arr_C[0],nr_rows_C * nr_cols_C * N_STI * N_BIN* sizeof(cuComplex),cudaMemcpyDeviceToHost); //d_c => h_arr_C[0]
//
//	std::cout << "C =" << std::endl;
//	print_matrix(h_C, nr_rows_C*N_STI, nr_cols_C*N_BIN);
//
//	//Free GPU memory
//	cudaFree(d_A);
//	cudaFree(d_B);
//	cudaFree(d_C);
//
//	// Destroy the handle
//	cublasDestroy(handle);
//
//	// Free CPU memory
//	free(h_A);
//	free(h_B);
//	free(h_C);
//
//	return 0;
//}


#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <curand.h>
#include <assert.h>
#include <unistd.h>
#include <cublas_v2.h>
#include <iostream>
#include <complex.h>
#include <math.h>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include "cublas_beamformer.h"

using namespace std;

// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on GPU
void GPU_fill(cuComplex *A, int nr_rows_A, int nr_cols_A) {
	cuComplex *G;
	G = new cuComplex[nr_rows_A*nr_cols_A];
	//		for(int j = 0; j < nr_cols_A; ++j){
	//			for(int i = 0; i < nr_rows_A; ++i){
	//				G[j * nr_rows_A + i].x = (j * nr_rows_A + i + 1)%(nr_rows_A*nr_cols_A/(N_STI*N_BIN));
	//				G[j * nr_rows_A + i].y = (j * nr_rows_A + i + 1)%(nr_rows_A*nr_cols_A/(N_STI*N_BIN));
	//			}
	//		}
	for(int i = 0; i < nr_rows_A*nr_cols_A; ++i){
		G[i].x = (i + 1)%(nr_rows_A*nr_cols_A/(N_BIN));
		G[i].y = (i + 1)%(nr_rows_A*nr_cols_A/(N_BIN));

	}

	cudaMemcpy(A,G,nr_rows_A * nr_cols_A * sizeof(cuComplex),cudaMemcpyHostToDevice);
	delete[] G;
}

void GPU_fill2(cuComplex *A, int nr_rows_A, int nr_cols_A) {
	cuComplex *G;
	G = new cuComplex[nr_rows_A*nr_cols_A];
	//	for(int j = 0; j < nr_cols_A; ++j){
	//		for(int i = 0; i < nr_rows_A; ++i){
	//			G[j * nr_rows_A + i].x = (j * nr_rows_A + i)%(nr_rows_A*nr_cols_A/(N_STI*N_BIN));
	//			G[j * nr_rows_A + i].y = (j * nr_rows_A + i)%(nr_rows_A*nr_cols_A/(N_STI*N_BIN));
	//		}
	//	}
	for(int i = 0; i < nr_rows_A*nr_cols_A; ++i){
		G[i].x = i%(nr_rows_A*nr_cols_A/(N_BIN));
		G[i].y = i%(nr_rows_A*nr_cols_A/(N_BIN));
	}

	cudaMemcpy(A,G,nr_rows_A * nr_cols_A * sizeof(cuComplex),cudaMemcpyHostToDevice);
	delete[] G;
}


//Print matrix A(nr_rows_A, nr_cols_A) storage in column-major format
//void print_matrix(const cuComplex *A, int nr_rows_A, int nr_cols_A) {
//		for(int i = 0; i < nr_rows_A; ++i){
//			for(int j = 0; j < nr_cols_A; ++j){
////				cout << A[j * nr_rows_A + i].x << "+" << A[j * nr_rows_A + i].y << "i" <<" ";
//				printf("%i,%i: %e + %e i\n",i,j,A[j * nr_rows_A + i].x, A[j * nr_rows_A + i].y);
//			}
////			cout << endl;
//		}
////		cout << endl;
////	for(int i = 0; i < nr_rows_A*nr_cols_A; ++i){
////		printf("%i,: %e + %e i\n",i,A[i].x, A[i].y);
////	}
//}
void print_matrix(const cuComplex *A, int nr_rows_A, int nr_cols_A, int nr_sheets_A) {
	for(int i = 0; i < nr_rows_A; ++i){
		for(int j = 0; j < nr_cols_A; ++j){
			for(int k = 0; k < nr_sheets_A; ++k){
				//				cout << A[j * nr_rows_A + i].x << "+" << A[j * nr_rows_A + i].y << "i" <<" ";
				printf("%i,%i,%i: %e + %e i\n",i,j,k,A[k*nr_rows_A*nr_cols_A + j * nr_rows_A + i].x, A[k*nr_rows_A*nr_cols_A + j * nr_rows_A + i].y);
			}
		}
		//			cout << endl;
	}
	//		cout << endl;
	//	for(int i = 0; i < nr_rows_A*nr_cols_A; ++i){
	//		printf("%i,: %e + %e i\n",i,A[i].x, A[i].y);
	//	}
}


void print_matrix2(const float *A, int nr_rows_A, int nr_cols_A) {
	//	for(int j = 0; j < nr_cols_A; ++j){
	//		for(int i = 0; i < nr_rows_A; ++i){
	//			//cout << A[j * nr_rows_A + i].x << "+" << A[j * nr_rows_A + i].y << "i" <<" ";
	//			printf("%i,%i: %e\n",i,j,A[j * nr_rows_A + i]);
	//		}
	//		cout << endl;
	//	}
	//	cout << endl;

	for(int i = 0; i < nr_rows_A*nr_cols_A; ++i){
		printf("%i,: %e\n",i,A[i]);
	}
}


void beamform(const cuComplex * d_A,
		const cuComplex * d_B,cublasHandle_t handle,
		cuComplex * d_C) {

	// Allocate 3 arrays on CPU
	cudaError_t cudaStat;

	int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;

	//	nr_rows_A = N_TIME_STI;
	//	nr_cols_A = N_ELE;
	//	nr_rows_B = N_ELE;
	//	nr_cols_B = N_BEAM;
	//	nr_rows_C = N_TIME_STI;
	//	nr_cols_C = N_BEAM;
	nr_rows_A = N_BEAM;//N_BEAM;
	nr_cols_A = N_ELE;//N_ELE*N_BIN;
	nr_rows_B = N_ELE;//N_ELE*N_BIN;
	nr_cols_B = N_TIME;//N_BIN;
	nr_rows_C = N_BEAM;//N_BEAM*N_BIN;
	nr_cols_C = N_TIME;//N_BIN;

	const cuComplex **h_arr_A = 0; const cuComplex **h_arr_B = 0; cuComplex **h_arr_C = 0;
	//New code ////////
	h_arr_A = (const cuComplex **)malloc(nr_rows_A * nr_cols_A *N_BIN*sizeof(const cuComplex*)); //N_TIME instead of N_BIN
	h_arr_B = (const cuComplex **)malloc(nr_rows_B * nr_cols_B *N_BIN*sizeof(const cuComplex*)); //N_TIME instead of N_BIN
	h_arr_C = (cuComplex **)malloc(nr_rows_C * nr_cols_C *N_BIN*sizeof(cuComplex*)); //N_TIME instead of N_BIN

	for(int i = 0; i < N_BIN; i++){ //N_TIME instead of N_BIN
		h_arr_A[i] = d_A + i*nr_rows_A*nr_cols_A;
		h_arr_B[i] = d_B + i*nr_rows_B*nr_cols_B;
		h_arr_C[i] = d_C + i*nr_rows_C*nr_cols_C;
	}

	//	delete[] d_A;
	//	delete[] d_B;

	//		cuComplex At[nr_rows_B*nr_cols_B];
	//		cudaMemcpy(At,h_arr_B[24],nr_rows_B*nr_cols_B*sizeof(cuComplex),cudaMemcpyDeviceToHost);
	//		cout << "At =" << endl;
	//		for(int i =0; i<nr_rows_B*nr_cols_B;i++)
	//			printf("%i,: %e + %e i\n",i,At[i].x, At[i].y);
	//			//cout << i << At[i].x << "+" << At[i].y << "i" <<" " ;

	cuComplex **d_arr_A = 0; cuComplex **d_arr_B = 0; cuComplex **d_arr_C = 0;
	cudaStat = cudaMalloc((void **)&d_arr_A,nr_rows_A * nr_cols_A * N_BIN * sizeof(cuComplex*)); //N_TIME instead of N_BIN
	assert(!cudaStat);
	cudaStat = cudaMalloc((void **)&d_arr_B,nr_rows_B * nr_cols_B * N_BIN * sizeof(cuComplex*)); //N_TIME instead of N_BIN
	assert(!cudaStat);
	cudaStat = cudaMalloc((void **)&d_arr_C,nr_rows_C * nr_cols_C * N_BIN * sizeof(cuComplex*)); //N_TIME instead of N_BIN
	assert(!cudaStat);

	cudaStat = cudaMemcpy(d_arr_A,h_arr_A,nr_rows_A * nr_cols_A * N_BIN * sizeof(cuComplex*),cudaMemcpyHostToDevice); //N_TIME instead of N_BIN
	assert(!cudaStat);
	cudaStat = cudaMemcpy(d_arr_B,h_arr_B,nr_rows_B * nr_cols_B * N_BIN * sizeof(cuComplex*),cudaMemcpyHostToDevice); //N_TIME instead of N_BIN
	assert(!cudaStat);
	cudaStat = cudaMemcpy(d_arr_C,h_arr_C,nr_rows_C * nr_cols_C * N_BIN * sizeof(cuComplex*),cudaMemcpyHostToDevice); //N_TIME instead of N_BIN
	assert(!cudaStat);

	//		cuComplex At[38*14];
	//		cudaMemcpy(At,h_arr_A[0],38*14*sizeof(cuComplex),cudaMemcpyDeviceToHost);

	int lda=nr_rows_A,ldb=nr_rows_B,ldc=nr_rows_C;
	cuComplex alf;
	cuComplex bet;

	alf.x = 1;
	alf.y = 0;
	bet.x = 0;
	bet.y = 0;
	//	const float *alpha = &alf;
	//	const float *beta = &bet;
	//New variables
	int batchCount = N_BIN; //N_TIME instead of N_BIN

	// Create a handle for CUBLAS
	cublasCreate(&handle);

	cublasStatus_t stat;
	// Do the actual multiplication
	//	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

	stat = cublasCgemmBatched(
			handle,
			CUBLAS_OP_N, // CUBLAS_OP_N,
			CUBLAS_OP_N,
			nr_rows_A,
			nr_cols_B,
			nr_cols_A,
			&alf,
			(const cuComplex **)d_arr_A,
			lda,
			(const cuComplex **)d_arr_B,
			ldb,
			&bet,
			(cuComplex **)d_arr_C,
			ldc,
			batchCount);


	if(stat != CUBLAS_STATUS_SUCCESS){
		cerr << "cublasCgemmBatched failed" << endl;
		exit(1);
	}
	assert(!cudaGetLastError());

	//////////////////////


	// Copy (and print) the result on host memory
	//	cuComplex *h_C = (cuComplex *)malloc(nr_rows_C * nr_cols_C * N_STI * N_BIN * sizeof(cuComplex));
	//	cudaMemcpy(h_C,h_arr_C[0],nr_rows_C * nr_cols_C * N_STI * N_BIN* sizeof(cuComplex),cudaMemcpyDeviceToHost); //d_c => h_arr_C[0]
	//
	//	cout << "C =" << endl;
	//	print_matrix(h_C, nr_rows_C*N_STI, nr_cols_C*N_BIN);

	//Free GPU memory
	//	cudaFree(d_A);
	//	cudaFree(d_B);
	//	cudaFree(d_C);

	// Destroy the handle
	//cublasDestroy(handle);

}

__global__
void sti_reduction(const cuComplex * beamformed,
		float * data_out) {

	int f = blockIdx.x;
	int b = blockIdx.y;
	int t = threadIdx.x;
	int s = blockIdx.z;

	int h = sample_idx(s*N_TIME_STI + t,b,f);
	int h1 = sample_idx(s*N_TIME_STI + t,b+N_BEAM1,f);

	float beam_power1;
	float beam_power2;
	float cross_power1;
	float cross_power2;

	cuFloatComplex samp1;
	cuFloatComplex samp2;
	float scale = 1.0/N_TIME_STI;

	//New variables//////
	__shared__ cuFloatComplex reduced_array1[N_STI_BLOC];
	__shared__ cuFloatComplex reduced_array[N_STI_BLOC];
	/////////////////////

	if (t < N_TIME_STI) {
		samp1.x = beamformed[h].x;
		samp1.y = beamformed[h].y;
		beam_power1 = (samp1.x * samp1.x) + (samp1.y * samp1.y);
		reduced_array[t].x = beam_power1;

		samp2.x = beamformed[h1].x;
		samp2.y = beamformed[h1].y;
		beam_power2 = (samp2.x * samp2.x) + (samp2.y * samp2.y);
		reduced_array[t].y = beam_power2;

		cross_power1 = (samp1.x * samp2.x) + (samp1.y * samp2.y);
		cross_power2 = (samp1.y * samp2.x) - (samp1.x * samp2.y);
		reduced_array1[t].x = cross_power1;
		reduced_array1[t].y = cross_power2;
	}

	//New code///////////////////////////////////////////////
	else{
		reduced_array[t].x = 0.0;
		reduced_array[t].y = 0.0;
		reduced_array1[t].x = 0.0;
		reduced_array1[t].y = 0.0;
	}
	__syncthreads();

	for(int k = blockDim.x/2; k>0; k>>=1){
		if(t<k){
			reduced_array[t].x += reduced_array[t+k].x;
			reduced_array[t].y += reduced_array[t+k].y;
			reduced_array1[t].x += reduced_array1[t+k].x;
			reduced_array1[t].y += reduced_array1[t+k].y;
		}
		__syncthreads();
	}

	if(t == 0){
		//New Code
		data_out[output_idx(0,b,s,f)] = reduced_array[0].x*scale; //x pol
		data_out[output_idx(1,b,s,f)] = reduced_array[0].y*scale; //y pol
		data_out[output_idx(2,b,s,f)] = reduced_array1[0].x*scale; //cross pol (x)
		data_out[output_idx(3,b,s,f)] = reduced_array1[0].y*scale;//cross pol (y)
	}
}

//#include <stdio.h>
//#include <stdlib.h>
//#include <cstdlib>
//#include <curand.h>
//#include <assert.h>
//#include <unistd.h>
//#include <cublas_v2.h>
//#include <iostream>
//#include <complex.h>
//#include <math.h>
//#include <cuComplex.h>
//#include <cuda_runtime.h>
//#include "cublas_beamformer.h"
//
//using namespace std;
//
//// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on GPU
//void GPU_fill(cuComplex *A, int nr_rows_A, int nr_cols_A) {
//
//	/*
//	// Create a pseudo-random number generator
//	curandGenerator_t prng;
//	curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
//
//	// Set the seed for the random number generator using the system clock
//	curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());
//
//	// Fill the array with random numbers on the device
//	curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);
//	*/
////	float *G;
//
//	cuComplex *G;
//	G = new cuComplex[nr_rows_A*nr_cols_A];
//	for(int i = 0; i < nr_rows_A*nr_cols_A; i++){
//		G[i].x = 1;
//		G[i].y = 0;
//	}
//	cudaMemcpy(A,G,nr_rows_A * nr_cols_A * sizeof(cuComplex),cudaMemcpyHostToDevice);
//	delete[] G;
//}
//
//
////Print matrix A(nr_rows_A, nr_cols_A) storage in column-major format
//void print_matrix(const cuComplex *A, int nr_rows_A, int nr_cols_A) {
//	for(int i = 0; i < nr_rows_A; ++i){
//		for(int j = 0; j < nr_cols_A; ++j){
//			std::cout << A[j * nr_rows_A + i].x + A[j * nr_rows_A + i].y << " ";
//		}
//		std::cout << std::endl;
//	}
//	std::cout << std::endl;
//}
//
////void beamform(const cuComplex * h_A,
////	const cuComplex * h_B,
////	cuComplex * h_C) { {
//int main(){
//	// Allocate 3 arrays on CPU
//	cudaError_t cudaStat;
//
//	int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;
//
// 	nr_rows_A = N_TIME_STI;
// 	nr_cols_A = N_ELE;
// 	nr_rows_B = N_ELE;
// 	nr_cols_B = N_BEAM;
// 	nr_rows_C = N_TIME_STI;
// 	nr_cols_C = N_BEAM;
//
//	// for simplicity we are going to use square arrays
//	//nr_rows_A = nr_cols_A = nr_rows_B = nr_cols_B = nr_rows_C = nr_cols_C = 3;
//
// 	cuComplex *h_A = (cuComplex *)malloc(nr_rows_A * nr_cols_A * N_STI * N_BIN * sizeof(cuComplex));
// 	cuComplex *h_B = (cuComplex *)malloc(nr_rows_B * nr_cols_B * N_STI * N_BIN * sizeof(cuComplex));
// 	cuComplex *h_C = (cuComplex *)malloc(nr_rows_C * nr_cols_C * N_STI * N_BIN * sizeof(cuComplex));
//
//	// Allocate 3 arrays on GPU
// 	cuComplex *d_A, *d_B, *d_C;
//	cudaMalloc(&d_A,nr_rows_A * nr_cols_A * N_STI * N_BIN * sizeof(cuComplex));
//	cudaMalloc(&d_B,nr_rows_B * nr_cols_B * N_STI * N_BIN * sizeof(cuComplex));
//	cudaMalloc(&d_C,nr_rows_C * nr_cols_C * N_STI * N_BIN * sizeof(cuComplex));
//
//	// Fill the arrays A and B on GPU with random numbers
//	GPU_fill(d_A, nr_rows_A*N_STI, nr_cols_A*N_BIN);
//	GPU_fill(d_B, nr_rows_B*N_STI, nr_cols_B*N_BIN);
//
//	// Optionally we can copy the data back on CPU and print the arrays
//	cudaMemcpy(h_A,d_A,nr_rows_A * nr_cols_A * N_STI * N_BIN * sizeof(cuComplex),cudaMemcpyDeviceToHost);
//	cudaMemcpy(h_B,d_B,nr_rows_B * nr_cols_B * N_STI * N_BIN * sizeof(cuComplex),cudaMemcpyDeviceToHost);
////	std::cout << "A =" << std::endl;
////	print_matrix(h_A, nr_rows_A*N_STI, nr_cols_A*N_BIN);
////	std::cout << "B =" << std::endl;
////	print_matrix(h_B, nr_rows_B*N_STI, nr_cols_B*N_BIN);
//
//	cuComplex **h_arr_A = 0; cuComplex **h_arr_B = 0; cuComplex **h_arr_C = 0;
//	//New code ////////
//	h_arr_A = (cuComplex **)malloc(nr_rows_A * nr_cols_A *N_STI*N_BIN*sizeof(cuComplex*));
//	h_arr_B = (cuComplex **)malloc(nr_rows_B * nr_cols_B *N_STI*N_BIN*sizeof(cuComplex*));
//	h_arr_C = (cuComplex **)malloc(nr_rows_C * nr_cols_C *N_STI*N_BIN*sizeof(cuComplex*));
//
//	for(int i = 0; i < N_STI*N_BIN; i++){
//		h_arr_A[i] = d_A + i*N_ELE*N_TIME_STI;
//		h_arr_B[i] = d_B + i*N_ELE*N_BEAM;
//		h_arr_C[i] = d_C + i*N_TIME_STI*N_BEAM;
//	}
//
////	float At[40*38];
////	cudaMemcpy(At,h_arr_B[1],38*7*sizeof(float),cudaMemcpyDeviceToHost);
////	std::cout << "At =" << std::endl;
////	for(int i =0; i<38*7;i++)
////		std::cout << i << ":\t" << At[i] << std::endl;
//
//	cuComplex **d_arr_A = 0; cuComplex **d_arr_B = 0; cuComplex **d_arr_C = 0;
//	cudaStat = cudaMalloc(&d_arr_A,nr_rows_A * nr_cols_A * N_STI * N_BIN * sizeof(cuComplex*));
//	assert(!cudaStat);
//	cudaStat = cudaMalloc(&d_arr_B,nr_rows_B * nr_cols_B * N_STI * N_BIN * sizeof(cuComplex*));
//	assert(!cudaStat);
//	cudaStat = cudaMalloc(&d_arr_C,nr_rows_C * nr_cols_C * N_STI * N_BIN * sizeof(cuComplex*));
//	assert(!cudaStat);
//
//	//cudaMemcpy(d_arr_A,h_arr_A[0],nr_rows_A * nr_cols_A * N_STI * N_BIN * sizeof(float*),cudaMemcpyHostToDevice);
//
//	//printf("H_arr_A %d",&h_arr_A[0]);
//
//	cudaStat = cudaMemcpy(d_arr_A,h_arr_A,nr_rows_A * nr_cols_A * N_STI * N_BIN * sizeof(cuComplex*),cudaMemcpyHostToDevice);
//	assert(!cudaStat);
//	cudaStat = cudaMemcpy(d_arr_B,h_arr_B,nr_rows_B * nr_cols_B * N_STI * N_BIN * sizeof(cuComplex*),cudaMemcpyHostToDevice);
//	assert(!cudaStat);
//	cudaStat = cudaMemcpy(d_arr_C,h_arr_C,nr_rows_C * nr_cols_C * N_STI * N_BIN * sizeof(cuComplex*),cudaMemcpyHostToDevice);
//	assert(!cudaStat);
//	// Multiply A and B on GPU
//
//	//gpu_blas_mmul(d_arr_A, d_arr_B, d_arr_C, nr_rows_A, nr_rows_B, nr_cols_A);
//	int lda=nr_rows_A,ldb=nr_rows_B,ldc=nr_rows_A;
//	cuComplex alf;
//	cuComplex bet;
//
//	alf.x = 1;
//	alf.y = 0;
//	bet.x = 0;
//	bet.y = 0;
////	const float *alpha = &alf;
////	const float *beta = &bet;
//	//New variables
//	int batchCount = N_STI*N_BIN;
//
//	// Create a handle for CUBLAS
//	cublasHandle_t handle;
//	cublasCreate(&handle);
//
//	cublasStatus_t stat;
//	// Do the actual multiplication
////	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
//
//	stat = cublasCgemmBatched(
//			handle,
//			CUBLAS_OP_N,
//			CUBLAS_OP_N,
//			nr_rows_A,
//			nr_cols_A,
//			nr_rows_B,
//			&alf,
//			(const cuComplex **)d_arr_A,
//			lda,
//			(const cuComplex **)d_arr_B,
//			ldb,
//			&bet,
//			(cuComplex **)d_arr_C,
//			ldc,
//			batchCount);
//
//
//	if(stat != CUBLAS_STATUS_SUCCESS){
//		cerr << "cublasSgemmBatched failed" << endl;
//		exit(1);
//	}
//	assert(!cudaGetLastError());
//
//	//////////////////////
//
//
//	// Multiply A and B on GPU
//
//	//gpu_blas_mmul(d_A, d_B, d_C, nr_rows_A, nr_rows_B, nr_cols_A);
//
//	// Copy (and print) the result on host memory
//	cudaMemcpy(h_C,h_arr_C[0],nr_rows_C * nr_cols_C * N_STI * N_BIN* sizeof(cuComplex),cudaMemcpyDeviceToHost); //d_c => h_arr_C[0]
//
//	std::cout << "C =" << std::endl;
//	print_matrix(h_C, nr_rows_C*N_STI, nr_cols_C*N_BIN);
//
//	//Free GPU memory
//	cudaFree(d_A);
//	cudaFree(d_B);
//	cudaFree(d_C);
//
//	// Destroy the handle
//	cublasDestroy(handle);
//
//	// Free CPU memory
//	free(h_A);
//	free(h_B);
//	free(h_C);
//
//	return 0;
//}


//#include <stdio.h>
//#include <stdlib.h>
//#include <cstdlib>
//#include <curand.h>
//#include <assert.h>
//#include <unistd.h>
//#include <cublas_v2.h>
//#include <iostream>
//#include <complex.h>
//#include <math.h>
//#include <cuComplex.h>
//#include <cuda_runtime.h>
//#include "cublas_beamformer.h"
//
//using namespace std;
//
//// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on GPU
//void GPU_fill(cuComplex *A, int nr_rows_A, int nr_cols_A) {
//	cuComplex *G;
//	G = new cuComplex[nr_rows_A*nr_cols_A];
//	//		for(int j = 0; j < nr_cols_A; ++j){
//	//			for(int i = 0; i < nr_rows_A; ++i){
//	//				G[j * nr_rows_A + i].x = (j * nr_rows_A + i + 1)%(nr_rows_A*nr_cols_A/(N_STI*N_BIN));
//	//				G[j * nr_rows_A + i].y = (j * nr_rows_A + i + 1)%(nr_rows_A*nr_cols_A/(N_STI*N_BIN));
//	//			}
//	//		}
//	for(int i = 0; i < nr_rows_A*nr_cols_A; ++i){
//		G[i].x = (i + 1)%(nr_rows_A*nr_cols_A/(N_STI*N_BIN));
//		G[i].y = (i + 1)%(nr_rows_A*nr_cols_A/(N_STI*N_BIN));
//
//	}
//
//	cudaMemcpy(A,G,nr_rows_A * nr_cols_A * sizeof(cuComplex),cudaMemcpyHostToDevice);
//	delete[] G;
//}
//
//void GPU_fill2(cuComplex *A, int nr_rows_A, int nr_cols_A) {
//	cuComplex *G;
//	G = new cuComplex[nr_rows_A*nr_cols_A];
//	//	for(int j = 0; j < nr_cols_A; ++j){
//	//		for(int i = 0; i < nr_rows_A; ++i){
//	//			G[j * nr_rows_A + i].x = (j * nr_rows_A + i)%(nr_rows_A*nr_cols_A/(N_STI*N_BIN));
//	//			G[j * nr_rows_A + i].y = (j * nr_rows_A + i)%(nr_rows_A*nr_cols_A/(N_STI*N_BIN));
//	//		}
//	//	}
//	for(int i = 0; i < nr_rows_A*nr_cols_A; ++i){
//		G[i].x = i%(nr_rows_A*nr_cols_A/(N_STI*N_BIN));
//		G[i].y = i%(nr_rows_A*nr_cols_A/(N_STI*N_BIN));
//	}
//
//	cudaMemcpy(A,G,nr_rows_A * nr_cols_A * sizeof(cuComplex),cudaMemcpyHostToDevice);
//	delete[] G;
//}
//
//
////Print matrix A(nr_rows_A, nr_cols_A) storage in column-major format
//void print_matrix(const cuComplex *A, int nr_rows_A, int nr_cols_A) {
//	//	for(int i = 0; i < nr_rows_A; ++i){
//	//		for(int j = 0; j < nr_cols_A; ++j){
//	//			cout << A[j * nr_rows_A + i].x << "+" << A[j * nr_rows_A + i].y << "i" <<" ";
//	//		}
//	//		cout << endl;
//	//	}
//	//	cout << endl;
//	for(int i = 0; i < nr_rows_A*nr_cols_A; ++i){
//		printf("%i,: %e + %e i\n",i,A[i].x, A[i].y);
//	}
//}
//
//
//void print_matrix2(const float *A, int nr_rows_A, int nr_cols_A) {
////	for(int j = 0; j < nr_cols_A; ++j){
////		for(int i = 0; i < nr_rows_A; ++i){
////			//cout << A[j * nr_rows_A + i].x << "+" << A[j * nr_rows_A + i].y << "i" <<" ";
////			printf("%i,%i: %e\n",i,j,A[j * nr_rows_A + i]);
////		}
////		cout << endl;
////	}
////	cout << endl;
//
//	for(int i = 0; i < nr_rows_A*nr_cols_A; ++i){
//		printf("%i,: %e\n",i,A[i]);
//	}
//}
//
//void beamform(const cuComplex * d_A,
//		const cuComplex * d_B,cublasHandle_t handle,
//		cuComplex * d_C) {
//
//	// Allocate 3 arrays on CPU
//	cudaError_t cudaStat;
//
//	int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;
//
////	nr_rows_A = N_TIME_STI;
////	nr_cols_A = N_ELE;
////	nr_rows_B = N_ELE;
////	nr_cols_B = N_BEAM;
////	nr_rows_C = N_TIME_STI;
////	nr_cols_C = N_BEAM;
//	nr_rows_A = N_BEAM;
//	nr_cols_A = N_ELE*N_BIN;
//	nr_rows_B = N_ELE*N_BIN;
//	nr_cols_B = N_BIN;
//	nr_rows_C = N_BEAM;
//	nr_cols_C = N_BIN;
//
//	const cuComplex **h_arr_A = 0; const cuComplex **h_arr_B = 0; cuComplex **h_arr_C = 0;
//	//New code ////////
//	h_arr_A = (const cuComplex **)malloc(nr_rows_A * nr_cols_A *N_TIME*sizeof(const cuComplex*)); //N_TIME instead of N_BIN
//	h_arr_B = (const cuComplex **)malloc(nr_rows_B * nr_cols_B *N_TIME*sizeof(const cuComplex*)); //N_TIME instead of N_BIN
//	h_arr_C = (cuComplex **)malloc(nr_rows_C * nr_cols_C *N_TIME*sizeof(cuComplex*)); //N_TIME instead of N_BIN
//
//	for(int i = 0; i < N_TIME; i++){ //N_TIME instead of N_BIN
//		h_arr_A[i] = d_A + i*nr_rows_A*nr_cols_A;
//		h_arr_B[i] = d_B + i*nr_rows_B*nr_cols_B;
//		h_arr_C[i] = d_C + i*nr_rows_C*nr_cols_C;
//	}
//
//	//	delete[] d_A;
//	//	delete[] d_B;
//
////		cuComplex At[38*14];
////		cudaMemcpy(At,h_arr_A[0],38*14*sizeof(cuComplex),cudaMemcpyDeviceToHost);
////		cout << "At =" << endl;
////		for(int i =0; i<38*14;i++)
////			printf("%i,: %e + %e i\n",i,At[i].x, At[i].y);
////			cout << i << At[i].x << "+" << At[i].y << "i" <<" " ;
//
//	cuComplex **d_arr_A = 0; cuComplex **d_arr_B = 0; cuComplex **d_arr_C = 0;
//	cudaStat = cudaMalloc((void **)&d_arr_A,nr_rows_A * nr_cols_A * N_TIME * sizeof(cuComplex*)); //N_TIME instead of N_BIN
//	assert(!cudaStat);
//	cudaStat = cudaMalloc((void **)&d_arr_B,nr_rows_B * nr_cols_B * N_TIME * sizeof(cuComplex*)); //N_TIME instead of N_BIN
//	assert(!cudaStat);
//	cudaStat = cudaMalloc((void **)&d_arr_C,nr_rows_C * nr_cols_C * N_TIME * sizeof(cuComplex*)); //N_TIME instead of N_BIN
//	assert(!cudaStat);
//
//	cudaStat = cudaMemcpy(d_arr_A,h_arr_A,nr_rows_A * nr_cols_A * N_TIME * sizeof(cuComplex*),cudaMemcpyHostToDevice); //N_TIME instead of N_BIN
//	assert(!cudaStat);
//	cudaStat = cudaMemcpy(d_arr_B,h_arr_B,nr_rows_B * nr_cols_B * N_TIME * sizeof(cuComplex*),cudaMemcpyHostToDevice); //N_TIME instead of N_BIN
//	assert(!cudaStat);
//	cudaStat = cudaMemcpy(d_arr_C,h_arr_C,nr_rows_C * nr_cols_C * N_TIME * sizeof(cuComplex*),cudaMemcpyHostToDevice); //N_TIME instead of N_BIN
//	assert(!cudaStat);
//
//	//		cuComplex At[38*14];
//	//		cudaMemcpy(At,h_arr_A[0],38*14*sizeof(cuComplex),cudaMemcpyDeviceToHost);
//
//	int lda=nr_rows_A,ldb=nr_rows_B,ldc=nr_rows_C;
//	cuComplex alf;
//	cuComplex bet;
//
//	alf.x = 1;
//	alf.y = 0;
//	bet.x = 0;
//	bet.y = 0;
//	//	const float *alpha = &alf;
//	//	const float *beta = &bet;
//	//New variables
//	int batchCount = N_TIME; //N_TIME instead of N_BIN
//
//	// Create a handle for CUBLAS
//	cublasCreate(&handle);
//
//	cublasStatus_t stat;
//	// Do the actual multiplication
//	//	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
//
//	stat = cublasCgemmBatched(
//			handle,
//			CUBLAS_OP_N, // CUBLAS_OP_N,
//			CUBLAS_OP_N,
//			nr_rows_A,
//			nr_cols_B,
//			nr_cols_A,
//			&alf,
//			(const cuComplex **)d_arr_A,
//			lda,
//			(const cuComplex **)d_arr_B,
//			ldb,
//			&bet,
//			(cuComplex **)d_arr_C,
//			ldc,
//			batchCount);
//
//
//	if(stat != CUBLAS_STATUS_SUCCESS){
//		cerr << "cublasCgemmBatched failed" << endl;
//		exit(1);
//	}
//	assert(!cudaGetLastError());
//
//	//////////////////////
//
//
//	// Copy (and print) the result on host memory
//	//	cuComplex *h_C = (cuComplex *)malloc(nr_rows_C * nr_cols_C * N_STI * N_BIN * sizeof(cuComplex));
//	//	cudaMemcpy(h_C,h_arr_C[0],nr_rows_C * nr_cols_C * N_STI * N_BIN* sizeof(cuComplex),cudaMemcpyDeviceToHost); //d_c => h_arr_C[0]
//	//
//	//	cout << "C =" << endl;
//	//	print_matrix(h_C, nr_rows_C*N_STI, nr_cols_C*N_BIN);
//
//	//Free GPU memory
//	//	cudaFree(d_A);
//	//	cudaFree(d_B);
//	//	cudaFree(d_C);
//
//	// Destroy the handle
//	//cublasDestroy(handle);
//
//}
//
//__global__
//void sti_reduction(const cuFloatComplex * beamformed,
//		float * data_out) {
//
//	//	int f = blockIdx.x;
//	//	int b = blockIdx.y;
//	//	int t = threadIdx.x;
//	//	int s = blockIdx.z;
//	//
//	//	float beam_power;
//	//	float scale = 1.0/N_TIME_STI;
//	//
//	//	//New variable//////
//	//	__shared__ float reduced_array[N_STI_BLOC];
//	//	/////////////////////cuComplex *h_C = (cuComplex *)malloc(nr_rows_C * nr_cols_C * N_STI * N_BIN * sizeof(cuComplex));
//	//	cudaMemcpy(h_C,h_arr_C[0],nr_rows_C * nr_cols_C * N_STI * N_BIN* sizeof(cuComplex),cudaMemcpyDeviceToHost); //d_c => h_arr_C[0]
//	//
//	//	cout << "C =" << endl;
//	//	pr
//	//
//	//	if (t < N_TIME_STI) {
//	//		cuFloatComplex samp = beamformed[sample_idx(s*N_TIME_STI+t,b,f)];
//	//		//beam_power = (float)(cuCmulf(beamformed[sample_idx(s*N_TIME_STI+t,b,f)], cuConjf(beamformed[sample_idx(s*N_TIME_STI+t,b,f)])).x);
//	//		beam_power = samp.x * samp.x + samp.y * samp.y;
//	//
//	//		//atomicAdd(&data_out[output_idx(b,s,f)], beam_power*scale);
//	//	}
//	//
//	//	//New code///////////////////////////////////////////////
//	//
//	//	if(t<N_TIME_STI){
//	//		reduced_array[t] = beam_power;
//	//	}
//	//	else{
//	//		reduced_array[t] = 0.0;
//	//	}
//	//	__syncthreads();
//	//
//	//	for(int k = blockDim.x/2; k>0; k>>=1){
//	//		if(t<k){
//	//			reduced_array[t] += reduced_array[t+k];
//	//		}
//	//		__syncthreads();
//	//	}
//	//	if(t == 0){
//	//		data_out[output_idx(b,s,f)] = reduced_array[0]*scale;
//	//	}
//
//	int f = blockIdx.x;
//	int b = blockIdx.y;
//	int t = threadIdx.x;
//	int s = blockIdx.z;
//
//	int h = sample_idx(t,b,f);//sample_idx(s*N_TIME_STI+t,b,f);
//
//	float beam_power1;
//	float beam_power2;
//	//	cuFloatComplex cross_pol;
//	//	cuFloatComplex samp1[N_TIME*N_BEAM*N_BIN];
//	//	cuFloatComplex samp2[N_TIME*N_BEAM*N_BIN];
//	cuFloatComplex samp1;
//	cuFloatComplex samp2;
//	float scale = 1.0/(N_TIME_STI*N_BIN); //Normalize the data replication by dividing by N_BIN
//
//	//New variables//////
//	__shared__ cuFloatComplex reduced_array1[N_STI_BLOC];
//	__shared__ cuFloatComplex reduced_array[N_STI_BLOC];
//	//__shared__ float y_pol[N_BEAM*N_BIN];
//	/////////////////////
//
//	if (t < N_TIME_STI) {
//		samp1.x = beamformed[h].x;
//		samp1.y = beamformed[h].y;
//		beam_power1 = (samp1.x * samp1.x) + (samp1.y * samp1.y);
//		reduced_array[t].x = beam_power1;
//
//		samp2.x = beamformed[h+(N_BEAM1*N_BIN*N_STI)].x; // Change made, multiplied by N_STI
//		samp2.y = beamformed[h+(N_BEAM1*N_BIN*N_STI)].y; // Change made, multiplied by N_STI
//		beam_power2 = (samp2.x * samp2.x) + (samp2.y * samp2.y);
//		reduced_array[t].y = beam_power2;
//
//		reduced_array1[t].x = (samp1.x * samp2.x) + (samp1.y * samp2.y);
//		reduced_array1[t].y = (samp1.y * samp2.x) - (samp1.x * samp2.y);
//	}
//
//	//New code///////////////////////////////////////////////
//	else{
//		reduced_array[t].x = 0.0;
//		reduced_array[t].y = 0.0;
//		reduced_array1[t].x = 0.0;
//		reduced_array1[t].y = 0.0;
//	}
//	__syncthreads();
//
//	for(int k = blockDim.x/2; k>0; k>>=1){
//		if(t<k){
//			reduced_array[t].x += reduced_array[t+k].x;
//			reduced_array[t].y += reduced_array[t+k].y;
//			reduced_array1[t].x += reduced_array1[t+k].x;
//			reduced_array1[t].y += reduced_array1[t+k].y;
//		}
//		__syncthreads();
//	}
//	if(t == 0){
//		//		data_out[output_idx(b,s,f)] = reduced_array[0].x*scale;
//
//		//New Code //Not this easy :( X  It is this easy :)
//		data_out[4*output_idx(b,s,f)] = reduced_array[0].x*scale; //x pol
//		data_out[4*output_idx(b,s,f)+1] = reduced_array[0].y*scale; //y pol
//		data_out[4*output_idx(b,s,f)+2] = reduced_array1[0].x*scale; //cross pol (x)
//		data_out[4*output_idx(b,s,f)+3] = reduced_array1[0].y*scale;//cross pol (y)
//	}
//
//
//	/////////////////////////////////////////////////////////
//}
