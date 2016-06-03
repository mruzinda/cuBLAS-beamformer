#include "cublas_beamformer.h"


#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <curand.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>

using namespace std;

void printUsage();

int main(int argc, char * argv[]) {
	// Parse input
	if (argc != 4) {
		printUsage();
		return -1;
	}
	char input_filename[128];
	char weight_filename[128];
	char output_filename[128];

	strcpy(input_filename,  argv[1]);
	strcpy(weight_filename, argv[2]);
	strcpy(output_filename, argv[3]);

	// File pointers
	FILE * data;
	FILE * weights;

	// File data pointers
	float * bf_data;
	float * bf_weights;

	// Complex data pointers
	float complex * data_dc;
	float complex * data_ch_form;
	float complex * weights_dc;
	float complex * weights_dc_n;

	// Allocate heap memory for file data
	bf_data = (float *)malloc(2*N_SAMP*sizeof(float));
	bf_weights = (float *)malloc(2*N_WEIGHTS*sizeof(float));
	data_dc = (float complex *)malloc(N_SAMP*sizeof(float complex *));
	data_ch_form = (float complex *)malloc(N_SAMP*sizeof(float complex *));
	weights_dc = (float complex *)malloc(N_WEIGHTS*sizeof(float complex *));
	weights_dc_n = (float complex *)malloc(N_WEIGHTS*sizeof(float complex *));
	//r_weights = (float complex *)malloc(N_WEIGHTS*N_TIME*sizeof(float complex *));

	// Open files
	data = fopen(input_filename, "r");
	weights = fopen(weight_filename, "r");

	// Read in data
	int j;
	if (data != NULL) {
		fread(bf_data, sizeof(float), 2*N_SAMP, data);
		// Make 'em complex!
		for (j = 0; j < N_SAMP; j++) {
			data_dc[j] = bf_data[2*j] + bf_data[(2*j)+1]*I;
		}

		//		// Specify grid and block dimensions
		//		dim3 dimBlock(N_STI_BLOC, 1, 1);
		//		dim3 dimGrid(N_BIN, N_BEAM1, N_STI);

		float complex data_r;
		int f = 0;
		if(f < N_SAMP){
			for (int c = 0; c < N_BIN; c++) {
				for(int d = 0; d < N_TIME; d++){
					for(int e = 0; e < N_ELE; e++){
						data_r = data_dc[d*N_BIN*N_ELE + c*N_ELE + e];
						data_ch_form[f] = data_r;
						f++;
					}
				}
			}
		}

		fclose(data);
	}
	free(bf_data);

	if (weights != NULL) {
		fread(bf_weights, sizeof(float), 2*N_WEIGHTS, weights);
		// Make 'em complex!
		//		for (u = 0; u < N_TIME; u++) {
		//			for(j = 0; j < N_WEIGHTS; j++){
		//				weights_dc[u*N_WEIGHTS+j] = bf_weights[2*j] + bf_weights[(2*j)+1]*I; //Removed conjugate
		//			}
		//		}

		for(j = 0; j < N_WEIGHTS; j++){
			weights_dc_n[j] = bf_weights[2*j] - bf_weights[(2*j)+1]*I; //Removed conjugate
		}

		int m,n;
		float complex transpose[N_BEAM][N_ELE*N_BIN];
		for(m=0;m<N_BEAM;m++){
			for(n=0;n<N_ELE*N_BIN;n++){
				transpose[m][n] = weights_dc_n[m*N_ELE*N_BIN + n];
			}
		}
		for(n=0;n<N_ELE*N_BIN;n++){
			for(m=0;m<N_BEAM;m++){
				weights_dc[n*N_BEAM+ m] = transpose[m][n];
			}
		}

		fclose(weights);
	}

	free(bf_weights);

//	for(int j = 0; j < N_SAMP/1000; j++)
//		printf("data_dc, %i:\t%.7e %.7ei \n",j,creal(data_ch_form[j]),cimag(data_ch_form[j]));
//			for(int i = 0; i < N_WEIGHTS; i++)
//				printf("weights_dc, %i: \t %.7e %.7ei\n",i,creal(weights_dc_n[i]),cimag(weights_dc_n[i]));
//	printf("data_dc_b,:\t%.7e %.7ei \n",creal(data_ch_form[(152000*25)-1]),cimag(data_ch_form[(152000*25)-1]));
//	printf("data_dc_a,:\t%.7e %.7ei \n",creal(data_ch_form[(152000*1)+1]),cimag(data_ch_form[(152000*1)+1]));


	// Allocate memory for the output
	float * output_f;
	//	output_f = (float *)calloc(N_OUTPUTS,sizeof(float));
	output_f = (float *)calloc(N_POL*(N_OUTPUTS/2),sizeof(float));

	struct timespec tstart = {0,0};
	struct timespec tstop  = {0,0};
	clock_gettime(CLOCK_MONOTONIC, &tstart);

	// Specify grid and block dimensions
	dim3 dimBlock(N_STI_BLOC, 1, 1);
	dim3 dimGrid(N_BIN, N_BEAM1, N_STI);

	cuComplex * d_data;
	cuComplex * d_weights;
	cuComplex * d_beamformed;//////////
	float * d_outputs;

	cudaMalloc((void **)&d_data, N_SAMP*sizeof(cuComplex)); //*N_BIN
	cudaMalloc((void **)&d_weights, N_WEIGHTS*sizeof(cuComplex)); //*N_TIME
	//cudaMalloc((void **)&d_outputs, N_OUTPUTS*sizeof(float));
	cudaMalloc((void **)&d_outputs, N_POL*(N_OUTPUTS*sizeof(float)/2));
	//	cudaError_t err_malloc = cudaMalloc((void **)&d_data, (N_SAMP + N_WEIGHTS)*sizeof(cuComplex) + N_OUTPUTS*sizeof(float));
	//	if (err_malloc != cudaSuccess) {
	//		printf("CUDA Error (cudaMalloc1): %s\n", cudaGetErrorString(err_malloc));
	//	}
	cudaError_t err_malloc = cudaMalloc((void **)&d_beamformed, N_TBF*sizeof(cuComplex));
	if (err_malloc != cudaSuccess) {
		printf("CUDA Error (cudaMalloc2): %s\n", cudaGetErrorString(err_malloc));
	}

	//	d_weights = d_data + N_SAMP;
	//	d_outputs = (float *)(d_data + N_SAMP + N_WEIGHTS);
	//	cudaMemset(d_outputs, 0.0, N_OUTPUTS*sizeof(float));

	//printf("data_dc weights_dc %.7e %e\n",data_dc,weights_dc);
	cudaMemcpy(d_data,    data_ch_form,   N_SAMP*sizeof(cuComplex), cudaMemcpyHostToDevice); //r_data instead of data_dc //*N_BIN
	cudaMemcpy(d_weights, weights_dc, N_WEIGHTS*sizeof(cuComplex), cudaMemcpyHostToDevice); //r_weights instead of weights_dc //*N_TIME


	//printf("data_dc:\t%.7e+%.7e*I\n weights_dc:\t%.7e+%.7e*I\n",data_dc[0],weights_dc[0]);

	// Run the beamformer
	//printf("D_data D_weights %.7e + %.7e*I\n",temp);


	// Allocate 3 arrays on GPU
	int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C, nr_sheets_C;

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
	nr_sheets_C = N_BIN;
//
//	cuComplex *b_A, *b_B, *b_C;
//	cudaMalloc((void **)&b_A,nr_rows_A * nr_cols_A * N_BIN * sizeof(cuComplex));
//	cudaMalloc((void **)&b_B,nr_rows_B * nr_cols_B * N_BIN * sizeof(cuComplex));
//	cudaMalloc((void **)&b_C,nr_rows_C * nr_cols_C * N_BIN * sizeof(cuComplex));
//
//	// Fill the arrays A and B on GPU with random numbers
//	GPU_fill(b_A, nr_rows_A, nr_cols_A*N_BIN);
//	GPU_fill2(b_B, nr_rows_B, nr_cols_B*N_BIN);
//
//	// Optionally we can copy the data back on CPU and print the arrays
//	cuComplex *h_A = (cuComplex *)malloc(nr_rows_A * nr_cols_A * N_BIN * sizeof(cuComplex));
//	cuComplex *h_B = (cuComplex *)malloc(nr_rows_B * nr_cols_B * N_BIN * sizeof(cuComplex));
//	cudaMemcpy(h_A,b_A,nr_rows_A * nr_cols_A * N_BIN * sizeof(cuComplex),cudaMemcpyDeviceToHost);
//	cudaMemcpy(h_B,b_B,nr_rows_B * nr_cols_B * N_BIN * sizeof(cuComplex),cudaMemcpyDeviceToHost);
//	//	cout << "A =" << endl;
//	//	print_matrix(h_A, nr_rows_A, nr_cols_A*N_BIN);
//	//	cout << "B =" << endl;
//	//	print_matrix(h_B, nr_rows_B, nr_cols_B*N_BIN);


	printf("Starting beamformer\n");
	cublasHandle_t handle;
	//	beamform(b_A, b_B, handle, d_beamformed);
	beamform(d_weights, d_data, handle, d_beamformed);//beamform<<<dimGrid, dimBlock>>>(d_data, d_weights, d_beamformed);

//		cuFloatComplex *h_C2 = (cuFloatComplex *)malloc(nr_rows_C * nr_cols_C * N_BIN * sizeof(cuFloatComplex)); // N_TIME instead of N_BIN
//		cudaMemcpy(h_C2,d_beamformed,nr_rows_C * nr_cols_C * N_BIN * sizeof(cuFloatComplex),cudaMemcpyDeviceToHost); // N_TIME instead of N_BIN
//
//		cout << "C =" << endl;
		//print_matrix(h_C2, 1, 1, 1); // N_TIME instead of N_BIN, but use 2 or 4 so it doesn't print for very long.
//		int my_bin = 3;
//		int my_beam = 12;
//		int my_time = 3500;
//		printf("%i,%i,%i: %e + %e i\n",my_time,my_beam,my_bin,h_C2[sample_idx(my_time,my_beam,my_bin)].x, h_C2[sample_idx(my_time,my_beam,my_bin)].y);

//	for(int u = 23; u<24; u++){
//		for(int w = 0; w<N_TIME; w++){
//			for(int q = 0; q<N_BEAM; q++){
//				printf("Freq %i, Time %i, Beam %i: %e + %e i\n",u,w,q,h_C2[sample_idx(w,q,u)].x, h_C2[sample_idx(w,q,u)].y);
//			}
//		}
//	}


	cudaError_t err_code = cudaGetLastError();
	if (err_code != cudaSuccess) {
		printf("CUDA Error (beamform): %s\n", cudaGetErrorString(err_code));
	}

	//printf("Beamformed %e+%e*I\n", temp);

	printf("Starting sti_reduction\n");
	sti_reduction<<<dimGrid, dimBlock>>>(d_beamformed,d_outputs);
	printf("Finishing sti_reduction\n");

	err_code = cudaGetLastError();
	if (err_code != cudaSuccess) {
		printf("CUDA Error (sti_reduction): %s\n", cudaGetErrorString(err_code));
	}


	//cudaMemcpy(output_f, d_outputs, N_OUTPUTS*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(output_f, d_outputs, N_POL*(N_OUTPUTS*sizeof(float)/2),cudaMemcpyDeviceToHost);

//	int my_bin = 1;
//	int my_beam = 2;
//	int my_time = 49;
//	int my_pol = 0;
//	printf("%i,%i,%i,%i: %e \n",my_pol,my_beam,my_bin,my_time,output_f[output_idx(my_pol,my_beam,my_time,my_bin)]);
//		cout << "C =" << endl;
//		print_matrix2(output_f, N_POL*(nr_rows_C*N_BIN/2), nr_cols_C);

	//	for(int u = 23; u<24; u++){
	//		for(int w = 0; w<N_TIME; w++){
	//			for(int q = 0; q<N_BEAM; q++){
	//				printf("Freq %i, Time %i, Beam %i: %e + %e i\n",u,w,q,h_C2[sample_idx(w,q,u)].x, h_C2[sample_idx(w,q,u)].y);
	//			}
	//		}
	//	}


	//printf("Output %e\n",output_f[0]);
	cudaFree(d_data);
	cudaFree(d_weights);
	cudaFree(d_outputs);

	clock_gettime(CLOCK_MONOTONIC, &tstop);
	//printf("Beamformer elapsed time: %.5f seconds\n",
	//((double)tstop.tv_sec + 1.0e-9*tstop.tv_nsec) -
	//((double)tstart.tv_sec + 1.0e-9*tstart.tv_nsec));

	// Save output data to file
	FILE * output;
	output = fopen(output_filename, "w");
	fwrite(output_f, sizeof(float), N_POL*(N_OUTPUTS/2), output);
	//	fwrite(output_f, sizeof(float), N_OUTPUTS, output);
	fclose(output);

	free(data_dc);
	free(weights_dc);
	free(output_f);
	//	cublasDestroy(handle);

	return 0;
}

void printUsage() {
	printf("Usage: my_beamformer <input_filename> <weight_filename> <output_filename>\n");
}

//For makefile at the very end "-fno-exceptions -fno-rtti"



//#include "cublas_beamformer.h"
//
//
//#include <stdio.h>
//#include <stdlib.h>
//#include <cstdlib>
//#include <curand.h>
//#include <assert.h>
//#include <unistd.h>
//#include <string.h>
//#include <stdlib.h>
//#include <time.h>
//#include <iostream>
//
//using namespace std;
//
//void printUsage();
//
//int main(int argc, char * argv[]) {
//	// Parse input
//	if (argc != 4) {
//		printUsage();
//		return -1;
//	}
//	char input_filename[128];
//	char weight_filename[128];
//	char output_filename[128];
//
//	strcpy(input_filename,  argv[1]);
//	strcpy(weight_filename, argv[2]);
//	strcpy(output_filename, argv[3]);
//
//	// File pointers
//	FILE * data;
//	FILE * weights;
//
//	// File data pointers
//	float * bf_data;
//	float * bf_weights;
//
//	// Complex data pointers
//	float complex * data_dc;
//	float complex * r_data;
//	float complex * weights_dc;
//	float complex * weights_dc_n;
//	float complex * r_weights;
//
//	// Allocate heap memory for file data
//	bf_data = (float *)malloc(2*N_SAMP*sizeof(float));
//	bf_weights = (float *)malloc(2*N_WEIGHTS*sizeof(float));
//	data_dc = (float complex *)malloc(N_SAMP*sizeof(float complex *));
//	r_data = (float complex *)malloc(N_SAMP*N_BIN*sizeof(float complex *));
//	weights_dc = (float complex *)malloc(N_WEIGHTS*sizeof(float complex *));
//	weights_dc_n = (float complex *)malloc(N_WEIGHTS*sizeof(float complex *));
//	r_weights = (float complex *)malloc(N_WEIGHTS*N_TIME*sizeof(float complex *));
//
//	// Open files
//	data = fopen(input_filename, "r");
//	weights = fopen(weight_filename, "r");
//
//	// Read in data
//	int j;
//	if (data != NULL) {
//		fread(bf_data, sizeof(float), 2*N_SAMP, data);
//		// Make 'em complex!
//		for (j = 0; j < N_SAMP; j++) {
//			data_dc[j] = bf_data[2*j] + bf_data[(2*j)+1]*I;
//		}
//
//		for (int c = 0; c < N_BIN; c++) {
//			for(int d = 0; d < N_SAMP; d++){
//				r_data[c*N_SAMP+d] = data_dc[d];
//			}
//		}
//
//		fclose(data);
//	}
//	free(bf_data);
//
//	if (weights != NULL) {
//		fread(bf_weights, sizeof(float), 2*N_WEIGHTS, weights);
//		// Make 'em complex!
//		//		for (u = 0; u < N_TIME; u++) {
//		//			for(j = 0; j < N_WEIGHTS; j++){
//		//				weights_dc[u*N_WEIGHTS+j] = bf_weights[2*j] + bf_weights[(2*j)+1]*I; //Removed conjugate
//		//			}
//		//		}
//
//		for(j = 0; j < N_WEIGHTS; j++){
//			weights_dc_n[j] = bf_weights[2*j] + bf_weights[(2*j)+1]*I; //Removed conjugate
//		}
//
//		int m,n;
//		float complex transpose[N_BEAM][N_ELE*N_BIN];
//		for(m=0;m<N_BEAM;m++){
//			for(n=0;n<N_ELE*N_BIN;n++){
//				transpose[m][n] = weights_dc_n[m*N_ELE*N_BIN + n];
//			}
//		}
//		for(n=0;n<N_ELE*N_BIN;n++){
//			for(m=0;m<N_BEAM;m++){
//				weights_dc[n*N_BEAM+ m] = transpose[m][n];
//			}
//		}
//
//		for (int u = 0; u < N_TIME; u++) {
//			for(j = 0; j < N_WEIGHTS; j++){
//				r_weights[u*N_WEIGHTS+j] = weights_dc[j];
//			}
//		}
//
//		fclose(weights);
//	}
//	free(bf_weights);
//
////		for(int j = 0; j < N_SAMP/1000; j++)
////			printf("data_dc, %i:\t%.7e %.7ei\t \n",j,creal(r_data[j+N_SAMP*3]),cimag(r_data[j+N_SAMP*3]));
//			for(int i = 0; i < N_WEIGHTS/2; i++)
//				printf("weights_dc, %i: \t %.7e %.7ei\n",i,creal(r_weights[i+N_WEIGHTS*50]),cimag(r_weights[i+N_WEIGHTS*50]));
//
//
//	// Allocate memory for the output
//	float * output_f;
//	//	output_f = (float *)calloc(N_OUTPUTS,sizeof(float));
//	output_f = (float *)calloc(N_POL*(N_OUTPUTS/2),sizeof(float));
//
//	struct timespec tstart = {0,0};
//	struct timespec tstop  = {0,0};
//	clock_gettime(CLOCK_MONOTONIC, &tstart);
//
//	// Specify grid and block dimensions
//	dim3 dimBlock(N_STI_BLOC, 1, 1);
//	dim3 dimGrid(N_BIN, N_BEAM1, N_STI);
//
//	cuComplex * d_data;
//	cuComplex * d_weights;
//	cuComplex * d_beamformed;//////////
//	float * d_outputs;
//
//	cudaMalloc((void **)&d_data, N_SAMP*N_BIN*sizeof(cuComplex)); //*N_BIN
//	cudaMalloc((void **)&d_weights, N_WEIGHTS*N_TIME*sizeof(cuComplex)); //*N_TIME
//	//cudaMalloc((void **)&d_outputs, N_OUTPUTS*sizeof(float));
//	cudaMalloc((void **)&d_outputs, N_POL*(N_OUTPUTS*sizeof(float)/2));
//	//	cudaError_t err_malloc = cudaMalloc((void **)&d_data, (N_SAMP + N_WEIGHTS)*sizeof(cuComplex) + N_OUTPUTS*sizeof(float));
//	//	if (err_malloc != cudaSuccess) {
//	//		printf("CUDA Error (cudaMalloc1): %s\n", cudaGetErrorString(err_malloc));
//	//	}
//	cudaError_t err_malloc = cudaMalloc((void **)&d_beamformed, N_TBF*sizeof(cuComplex));
//	if (err_malloc != cudaSuccess) {
//		printf("CUDA Error (cudaMalloc2): %s\n", cudaGetErrorString(err_malloc));
//	}
//
//	//	d_weights = d_data + N_SAMP;
//	//	d_outputs = (float *)(d_data + N_SAMP + N_WEIGHTS);
//	//	cudaMemset(d_outputs, 0.0, N_OUTPUTS*sizeof(float));
//
//	//printf("data_dc weights_dc %.7e %e\n",data_dc,weights_dc);
//	cudaMemcpy(d_data,    r_data,   N_SAMP*N_BIN*sizeof(cuComplex), cudaMemcpyHostToDevice); //r_data instead of data_dc //*N_BIN
//	cudaMemcpy(d_weights, r_weights, N_WEIGHTS*N_TIME*sizeof(cuComplex), cudaMemcpyHostToDevice); //r_weights instead of weights_dc //*N_TIME
//
//
//	//printf("data_dc:\t%.7e+%.7e*I\n weights_dc:\t%.7e+%.7e*I\n",data_dc[0],weights_dc[0]);
//
//	// Run the beamformer
//	//printf("D_data D_weights %.7e + %.7e*I\n",temp);
//
//
//	// Allocate 3 arrays on GPU
//	int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;
//
//	//	nr_rows_A = N_TIME_STI;
//	//	nr_cols_A = N_ELE;
//	//	nr_rows_B = N_ELE;
//	//	nr_cols_B = N_BEAM;
//	//	nr_rows_C = N_TIME_STI;
//	//	nr_cols_C = N_BEAM;
//
//	nr_rows_A = N_BEAM;
//	nr_cols_A = N_ELE*N_BIN;
//	nr_rows_B = N_ELE*N_BIN;
//	nr_cols_B = N_BIN;
//	nr_rows_C = N_BEAM;
//	nr_cols_C = N_BIN;
//
//	cuComplex *b_A, *b_B, *b_C;
//	cudaMalloc((void **)&b_A,nr_rows_A * nr_cols_A * N_BIN * sizeof(cuComplex));
//	cudaMalloc((void **)&b_B,nr_rows_B * nr_cols_B * N_BIN * sizeof(cuComplex));
//	cudaMalloc((void **)&b_C,nr_rows_C * nr_cols_C * N_BIN * sizeof(cuComplex));
//
//	// Fill the arrays A and B on GPU with random numbers
//	GPU_fill(b_A, nr_rows_A, nr_cols_A*N_BIN);
//	GPU_fill2(b_B, nr_rows_B, nr_cols_B*N_BIN);
//
//	// Optionally we can copy the data back on CPU and print the arrays
//	cuComplex *h_A = (cuComplex *)malloc(nr_rows_A * nr_cols_A * N_BIN * sizeof(cuComplex));
//	cuComplex *h_B = (cuComplex *)malloc(nr_rows_B * nr_cols_B * N_BIN * sizeof(cuComplex));
//	cudaMemcpy(h_A,b_A,nr_rows_A * nr_cols_A * N_BIN * sizeof(cuComplex),cudaMemcpyDeviceToHost);
//	cudaMemcpy(h_B,b_B,nr_rows_B * nr_cols_B * N_BIN * sizeof(cuComplex),cudaMemcpyDeviceToHost);
//	//	cout << "A =" << endl;
//	//	print_matrix(h_A, nr_rows_A, nr_cols_A*N_BIN);
//	//	cout << "B =" << endl;
//	//	print_matrix(h_B, nr_rows_B, nr_cols_B*N_BIN);
//
//
//	printf("Starting beamformer\n");
//	cublasHandle_t handle;
//	//	beamform(b_A, b_B, handle, d_beamformed);
//	beamform(d_weights, d_data, handle, d_beamformed);//beamform<<<dimGrid, dimBlock>>>(d_data, d_weights, d_beamformed);
//
////	cuFloatComplex *h_C2 = (cuFloatComplex *)malloc(nr_rows_C * nr_cols_C * N_BIN * sizeof(cuFloatComplex)); // N_TIME instead of N_BIN
////	cudaMemcpy(h_C2,d_beamformed,nr_rows_C * nr_cols_C * N_BIN * sizeof(cuFloatComplex),cudaMemcpyDeviceToHost); // N_TIME instead of N_BIN
////
////	cout << "C =" << endl;
////	print_matrix(h_C2, nr_rows_C, nr_cols_C*2); // N_TIME instead of N_BIN, but use 2 or 4 so it doesn't print for very long.
//
//	cudaError_t err_code = cudaGetLastError();
//	if (err_code != cudaSuccess) {
//		printf("CUDA Error (beamform): %s\n", cudaGetErrorString(err_code));
//	}
//
//	//printf("Beamformed %e+%e*I\n", temp);
//
//	printf("Starting sti_reduction\n");
//	sti_reduction<<<dimGrid, dimBlock>>>(d_beamformed,d_outputs);
//	printf("Finishing sti_reduction\n");
//
//	err_code = cudaGetLastError();
//	if (err_code != cudaSuccess) {
//		printf("CUDA Error (sti_reduction): %s\n", cudaGetErrorString(err_code));
//	}
//
//
//	//cudaMemcpy(output_f, d_outputs, N_OUTPUTS*sizeof(float), cudaMemcpyDeviceToHost);
//	cudaMemcpy(output_f, d_outputs, N_POL*(N_OUTPUTS*sizeof(float)/2),cudaMemcpyDeviceToHost);
//
//	//	cout << "C =" << endl;
//	//	print_matrix2(output_f, N_STI, N_POL*(nr_cols_C*N_BIN/2));
//
//	//printf("Output %e\n",output_f[0]);
//	cudaFree(d_data);
//	cudaFree(d_weights);
//	cudaFree(d_outputs);
//
//	clock_gettime(CLOCK_MONOTONIC, &tstop);
//	//printf("Beamformer elapsed time: %.5f seconds\n",
//	//((double)tstop.tv_sec + 1.0e-9*tstop.tv_nsec) -
//	//((double)tstart.tv_sec + 1.0e-9*tstart.tv_nsec));
//
//	// Save output data to file
//	FILE * output;
//	output = fopen(output_filename, "w");
//	fwrite(output_f, sizeof(float), N_POL*(N_OUTPUTS/2), output);
//	//	fwrite(output_f, sizeof(float), N_OUTPUTS, output);
//	fclose(output);
//
//	free(data_dc);
//	free(weights_dc);
//	free(output_f);
//	//	cublasDestroy(handle);
//
//	return 0;
//}
//
//void printUsage() {
//	printf("Usage: my_beamformer <input_filename> <weight_filename> <output_filename>\n");
//}
//
////For makefile at the very end "-fno-exceptions -fno-rtti"
