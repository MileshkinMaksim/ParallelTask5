#include <iostream>
#include <cstring>
#include <cmath>
#include <ctime>
#include <iomanip>


#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <mpi.h>

#define ABS(X) ((X) < 0 ? -1 * (X) : (X))
#define MAX(X, Y) ((X) > (Y) ? (X) : (Y))
#define MIN(X, Y) ((X) > (Y) ? (Y) : (X))

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#define mpiErrchk(ans) { mpiAssert((ans), __FILE__, __LINE__); }
inline void mpiAssert(mpiError_t code, const char* file, int line, bool abort = true)
{
    if (code != MPI_SUCCESS)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// pointers for error and other matrixes
double*	A 		    = nullptr,
	    Anew	    = nullptr,  
	    A_d 	    = nullptr,  
	    Anew_d	    = nullptr,  
        buff        = nullptr,  // buffer for abs_diff calculation
	    d_out 		= nullptr,  // buffer for error on device
	    d_temp_storage = nullptr; 

// handler funnction which executes before end of program execution and frees memory allocated dynamically
void free_pointers()
{
	if (A) 	 	gpuErrchk(cudaFreeHost(A))
	if (Anew)	gpuErrchk(cudaFreeHost(Anew))
	if (A_d)	gpuErrchk(cudaFree(A_d))
	if (Anew_d)	gpuErrchk(cudaFree(Anew_d))
    if (buff)	gpuErrchk(cudaFree(buff))
	if (d_out)	gpuErrchk(cudaFree(d_out))
	if (d_temp_storage)gpuErrchk(cudaFree(d_temp_storage))
}

__global__ void iterateMatrix(double* A, double* Anew, size_t netSize, size_t intervalOneDevice)
{
	unsigned int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y_idx = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int index = y_idx * netSize + x_idx;

	if(x_idx < 1 || y_idx < 2 || x_idx > netSize - 2 || y_idx > intervalOneDevice - 2) return;
	Anew[index] = 0.25 * (
		A[index - 1] + 
		A[index - netSize] + 
		A[index + netSize] + 
		A[index + 1]
	);
}

__global__ void initMatrix(
    double* A, double* Anew,
    int netSize, double hst, double hsb, double vsl, double vsr,
    double tl, double tr, double bl, double br)
{
    //this functions initializes matrix borders in O(n)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 0 || i >= netSize*netSize) return;
    A[i * netSize] = vsl * i + tl;
    A[i] = hst * i + tl;
    A[((netSize - 1) - i) * netSize + (netSize - 1)] = vsr * i + br;
    A[(netSize - 1) * netSize + ((netSize - 1) - i)] = hsb * i + br;

    Anew[i * netSize] = vsl * i + tl;
    Anew[i] = hst * i + tl;
    Anew[((netSize - 1) - i) * netSize + (netSize - 1)] = vsr * i + br;
    Anew[(netSize - 1) * netSize + ((netSize - 1) - i)] = hsb * i + br;
}

// interpolation on the matrix edges between devices
__global__ void iterate_boundaries(double* A, double* Anew, size_t netSize, size_t intervalOneDevice){
	unsigned int up_idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int down_idx = up_idx;

	// check if horizontal index between 1 and (size - 2) then calculates result
	if (!(up_idx == 0 || up_idx > netSize - 2)) {
		Anew[netSize + up_idx] = 0.25 * (
			A[netSize + up_idx - 1] + A[j] + 
			A[2 * netSize + up_idx] + 
			A[netSize + up_idx + 1]
		);
		int penultElement = intervalOneDevice - 2;
		Anew[penultElement * netSize + down_idx] = 0.25 * (
			A[penultElement * netSize + down_idx - 1] + 
			A[(penultElement - 1) * netSize + down_idx] + 
			A[(penultElement + 1) * netSize + down_idx] + 
			A[penultElement * netSize + down_idx + 1]
		);
	}
}

// absolute difference between A and Anew stored in buff
__global__ void abs_diff(double* A, double* Anew, double* buff, size_t netSize, size_t intervalOneDevice) {

	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	size_t index = y * netSize + x;
	
	// check if idx is in allocated area then calculate result
	if(!(x <= 0 || y <= 0 || x >= (netSize - 1) || y >= (intervalOneDevice - 1)))
	{
		buff[index] = std::abs(A[index] - Anew[index]);
	}
}

int findNearestPowerOfTwo(size_t num) {
    int power = 1;
    while (power < num) {
        power <<= 1;
    }
    return power;
}

int main(int argc, char* argv[])
{
	auto atExitStatus = std::atexit(free_pointers);
	if (atExitStatus != 0)
	{
		std::cout << "Register error" << std::endl;
		exit(-1);
	}

    int netSize = 128;
    double minError = 0.000001;
    int maxIterations = 0;
    int toPrintResult = 0;
    char* end;

    //correct input check
    if (argc != 5) {
        std::cout << "You must enter exactly 4 arguments:\n1. Grid size (one number)\n2. Minimal error\n3. Iterations amount\n4. Print final result (1 - Yes/0 - No)\n";
        return -1;
    }
    else {
        netSize = strtol(argv[1], &end, 10);
        minError = strtod(argv[2], &end);
        maxIterations = strtol(argv[3], &end, 10);
        toPrintResult = strtol(argv[4], &end, 10);
    }
    std::cout << netSize << " " << minError << " " << maxIterations << std::endl;
	
		int totalNetSize = netSize * netSize;  // total matrix size

		// rank - number of device, deviceGroupSize - number of devices used by MPI, errCode- buffer for error message
		int rank, deviceGroupSize, errCode;

		if ((errCode = MPI_Init(&Argc, &argv)) != 0) return errCode;

		mpiErrchk(MPI_Comm_rank(MPI_COMM_WORLD, &rank))

		mpiErrchk(MPI_Comm_size(MPI_COMM_WORLD, &deviceGroupSize))

		// check if programm uses enough number of devices for next calculations
		int deviceCount = 0;
		cudaGetDeviceCount(&deviceCount);
		printf("%d - number of devices\n", deviceCount);
		if (deviceCount < deviceGroupSize || deviceGroupSize < 1) {
			std::cout << "Invalid number of devices!";
			std::exit(-1);
		}

		// choose device
		gpuErrchk(cudaSetDevice(rank))
		printf("device rank: %d\n", rank);

		// edges for calculating
		size_t rowsForOneProcess = netSize / deviceGroupSize;
		size_t start_y_idx = rowsForOneProcess * rank;

		//allocate matrices on host
		gpuErrchk(cudaMallocHost((void**)&A, totalNetSize * sizeof(double)))
		gpuErrchk(cudaMallocHost((void**)&Anew, totalNetSize * sizeof(double)))

		std::memset(A, 0, totalNetSize * sizeof(double));
		std::memset(Anew, 0, totalNetSize * sizeof(double));


		// matrix edge interpolation

	//values of net edges
    const double tl = 10, //top left
        tr = 20, //top right
        bl = 20, //bottom left
        br = 30; //bottom right

    const double hst = (tr - tl) / (netSize - 1), //horizontal step top
        hsb = (bl - br) / (netSize - 1), //horizontal step bottom
        vsl = (bl - tl) / (netSize - 1), //vertical step left
        vsr = (tr - br) / (netSize - 1); //vertical step right

    	//initialising A_d and Anew (device)
    	initMatrix <<< MAX((int)(netSize / threadsPerBlock.x), 1), MIN(threadsPerBlock.x, netSize) >>> (A_d, Anew_d, netSize, hst, hsb, vsl, vsr, tl, tr, bl, br);
    	gpuErrchk( cudaGetLastError(), A_h, Anew, A_d, max );

		// calculate used area for each process
		if (rank != 0 && rank != deviceGroupSize - 1)
		{
			rowsForOneProcess += 2;
		}
		else if (deviceGroupSize != 1)
		{
			rowsForOneProcess += 1;
		}

		// memory size for one device
		size_t alloc_memsize = netSize * rowsForOneProcess;

		// memory allocation for pointer will be used on device
		gpuErrchk(cudaMalloc((void**)&buff, alloc_memsize * sizeof(double)))
		gpuErrchk(cudaMalloc((void**)&A_d, alloc_memsize * sizeof(double)))
		gpuErrchk(cudaMalloc((void**)&Anew_d, alloc_memsize * sizeof(double)))

		// memset + memcpy
		size_t offset = (rank != 0) ? size : 0;
		gpuErrchk(cudaMemset(A_d, 0, sizeof(double) * alloc_memsize))
		gpuErrchk(cudaMemset(Anew_d, 0, sizeof(double) * alloc_memsize))
		gpuErrchk(cudaMemcpy(A_d, A + (start_y_idx * size) - offset, sizeof(double) * alloc_memsize, cudaMemcpyHostToDevice))
		gpuErrchk(cudaMemcpy(Anew_d, Anew + (start_y_idx * size) - offset, sizeof(double) * alloc_memsize, cudaMemcpyHostToDevice))

		// allocates buffer 'd_out' to contain max('abs_diff' function result)
		double* d_out;
		gpuErrchk(cudaMalloc((void**)&d_out, sizeof(double)))

		// allocates memory for temporary storage ton use Max reduction and sets temp_storage_bytes with size of d_temp_storage in bytes
		void* d_temp_storage = NULL;
		size_t temp_storage_bytes = 0;
		gpuErrchk(cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, buff, d_out, netSize * rowsForOneProcess))
		gpuErrchk(cudaMalloc(&d_temp_storage, temp_storage_bytes))

		// variables for loop execution
		double curError = minError + 1.0;  // current error
		int iteration = 0;  // current number of iterations

		// streams for calculations: cuda_stream - for blocks to sync them, matrix_calc_stream - for other operation
		cudaStream_t cuda_stream, matrix_calc_stream;
		gpuErrchk(cudaStreamCreate(&cuda_stream))
		gpuErrchk(cudaStreamCreate(&matrix_calc_stream))

		// params for cuda functions
		unsigned int threads_x = std::min(findNearestPowerOfTwo(netSize), 1024);
		unsigned int blocks_y = rowsForOneProcess;
		unsigned int blocks_x = netSize / threads_x + 1;

		dim3 blockDim(threads_x, 1);
		dim3 gridDim(blocks_x, blocks_y);

		while (iteration < maxIterations && curError > minError) {

			iterate_boundaries<<<netSize, 1, 0, cuda_stream>>>(A_d, Anew_d, netSize, rowsForOneProcess);

			iterateMatrix<<<gridDim, blockDim, 0, matrix_calc_stream>>>(A_d, Anew_d, netSize, rowsForOneProcess);
			
			// updates curError 1/100 times of main cycle iterations and on the last iteration
			if (iteration % 100 == 0 || iteration + 1 == maxIterations) {
				
				// synchronize to understand either we can make operations with matrix or not
				gpuErrchk(cudaStreamSynchronize(cuda_stream))

				abs_diff<<<gridDim, blockDim, 0, matrix_calc_stream>>>(A_d, Anew_d, buff, netSize, rowsForOneProcess);

				// cub max reduction
				gpuErrchk(cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, buff, d_out, alloc_memsize, matrix_calc_stream))

				// synchronize streams to receive actual d_out max values from all devices
				gpuErrchk(cudaStreamSynchronize(matrix_calc_stream))

				// receive max d_out values from all devices
				mpiErrchk(MPI_Allreduce((void*)d_out, (void*)d_out, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD))

				// copy values from d_out on GPU to curError on CPU
				gpuErrchk(cudaMemcpyAsync(&curError, d_out, sizeof(double), cudaMemcpyDeviceToHost, matrix_calc_stream);)
			}

			gpuErrchk(cudaStreamSynchronise(cuda_stream))
			// receive top edge
			if (rank != 0)
			{
				mpiErrchk(MPI_Sendrecv( Anew_d + netSize + 1, netSize - 2, MPI_DOUBLE, rank - 1, 0, Anew_d + 1, netSize - 2, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE))
			}

			// receive bottom edge
			if (rank != deviceGroupSize - 1)
			{
				mpiErrchk(MPI_Sendrecv(Anew_d + (rowsForOneProcess - 2) * netSize + 1, netSize - 2, MPI_DOUBLE, rank + 1, 0, Anew_d + (rowsForOneProcess - 1) * netSize + 1, netSize - 2, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE))
			}

			// synchronize streams before next calculations
			gpuErrchk(cudaStreamSynchronize(matrix_calc_stream))

			++iteration;
			std::swap(A_d, Anew_d); // swap pointers for next calculations
		}

		gpuErrchk(cudaStreamDestroy(cuda_stream))
		gpuErrchk(cudaStreamDestroy(matrix_calc_stream))

		if (rank == 0) {
			printf("Iterations: %d\nAccuracy: %lf\n", iteration, curError);
		}
		
		// end MPI engine
		mpiErrchk(MPI_Finalize())

		std::cout << "MPI engine was shut down" << std::endl;

	return 0;
}