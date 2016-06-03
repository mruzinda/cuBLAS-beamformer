#ifndef CUBLAS_BEAMFORMER
#define CUBLAS_BEAMFORMER

// beamformer_gpu.h

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <cublas_v2.h>
#include <cuComplex.h>

#define N_ELE	   38	// Number of elements/antennas in the array
#define N_BIN	   25	// Number of frequency bins
#define N_TIME	   4000	//40 // Number of decimated time samples
#define N_BEAM     280   // Number of beams we are forming
#define N_POL       4
#define N_BEAM1    (N_BEAM/2)   // Number of beams we are forming
#define N_TIME_STI 40	//40 // Number of decimated time samples per integrated beamformer output
#define N_STI	   (N_TIME/N_TIME_STI) // Number of short time integrations
#define N_STI_BLOC 64
#define N_ELE_BLOC 64
#define N_SAMP     (N_ELE*N_BIN*N_TIME) // Number of complex samples to process
#define N_WEIGHTS  (N_ELE*N_BIN*N_BEAM) // Number of complex beamformer weights
#define N_OUTPUTS  (N_BEAM*N_STI*N_BIN) // Number of complex samples in output structure
//
#define N_TBF     (N_BEAM*N_BIN*N_TIME)
//

#define input_idx(t,f,e)     ((e) + (f)*N_ELE + (t)*N_ELE*N_BIN)
#define weight_idx(b,f,e)    ((e) + (f)*N_ELE + (b)*N_ELE*N_BIN)
//#define sample_idx(t,b,f)    (f + b*N_BIN + (t)*N_BEAM*N_BIN)
#define sample_idx(t,b,f)    ((b) + (t)*N_BEAM + (f)*N_BEAM*N_TIME)
//#define output_idx(b,s,f)    ((f) + (s)*N_BIN + (b)*N_BIN*N_STI)
//#define output_idx(b,s,f)    ((b) + (f)*N_BEAM1 + (s)*N_BEAM1*N_BIN)
#define output_idx(p,b,s,f)    ((b) + (p)*N_BEAM1 + (f)*N_BEAM1*N_POL + (s)*N_BEAM1*N_BIN*N_POL)

void print_matrix(const cuComplex *A, int nr_rows_A, int nr_cols_A, int nr_sheets_A);

void print_matrix2(const float *A, int nr_rows_A, int nr_cols_A);

void GPU_fill(cuComplex *A, int nr_rows_A, int nr_cols_A);

void GPU_fill2(cuComplex *A, int nr_rows_A, int nr_cols_A);

void beamform(const cuComplex * d_A,
	const cuComplex * d_B,cublasHandle_t handle,
	cuComplex * d_C);

__global__
void sti_reduction(const cuComplex * beamformed,
	               float * data_out);

#endif
