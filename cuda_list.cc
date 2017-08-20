#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

using namespace std;

int getNumDevices();
void listDevices();

#define CUDA_SAFE_CALL(call)                                          \
do {                                                                  \
	cudaError_t err = call;                                           \
	if (cudaSuccess != err) {                                         \
		fprintf(stderr, "Cuda error in func '%s' at line %i : %s.\n", \
		         __FUNCTION__, __LINE__, cudaGetErrorString(err) );   \
		exit(EXIT_FAILURE);                                           \
	}                                                                 \
} while (0)

void listDevices() {
	try {
		string outString;
		int numDevices = getNumDevices();
		for (int i = 0; i < numDevices; ++i) {
			cudaDeviceProp props;
			CUDA_SAFE_CALL(cudaGetDeviceProperties(&props, i));

			outString += to_string(i) + ";\t" + string(props.name) + ";";
			outString += "\tCC" + to_string(props.major) + to_string(props.minor) + ";";
			outString += "\t" + to_string(int(round(props.totalGlobalMem / (1024*1024*1024.0)))) + "GB\n";
		}
  cout << outString;
	} catch(std::runtime_error const& err) {
		std::cerr << "CUDA error: " << err.what() << '\n';
	}
}

int getNumDevices() {
	int deviceCount = -1;
	cudaError_t err = cudaGetDeviceCount(&deviceCount);
	if (err == cudaSuccess) {
		return deviceCount;
  }

	if (err == cudaErrorInsufficientDriver) {
		int driverVersion = -1;
		cudaDriverGetVersion(&driverVersion);
		if (driverVersion == 0) {
			throw std::runtime_error{"No CUDA driver found"};
    }
		throw std::runtime_error{"Insufficient CUDA driver: " + std::to_string(driverVersion)};
	}

	throw std::runtime_error{cudaGetErrorString(err)};
}

void usage() {
  cout << "Lists CUDA devices.\nFORMAT: deviceID;\tdeviceName;\tcompute capability version;\tsize of ram\n";
}

int main(int argc, char *argv[]) { 
  if (argc > 1) {
    usage();
  }
  listDevices();
  return 0;
}
