#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include "nvml.h"

using namespace std;

int getNumDevices();
void listDevices();


inline void throwString(std::string fileName, std::string lineNumber, std::string failedExpression) {
	throw "Check failed: " + failedExpression + ", file " + fileName + ", line " + lineNumber;
}

inline void throwString(std::string fileName, std::string lineNumber, std::string failedExpression, std::string comment) {
	throw "Check failed: " + failedExpression + " (" + comment + "), file " + fileName + ", line " + lineNumber;
}

#ifdef QUOTE
#error QUOTE macro defined not only in check.h
#endif

#ifdef QUOTE_VALUE
#error QUOTE_VALUE macro defined not only in check.h
#endif

#ifdef check
#error check macro defined not only in check.h
#endif

#define QUOTE(x) #x

#define QUOTE_VALUE(x) QUOTE(x)

#define check(expression, ...) \
{ \
	if (!(expression)) { \
		throwString(__FILE__, QUOTE_VALUE(__LINE__), #expression, ##__VA_ARGS__); \
	} \
}

#define CUDA_SAFE_CALL(call)                                          \
do {                                                                  \
	cudaError_t err = call;                                           \
	if (cudaSuccess != err) {                                         \
		fprintf(stderr, "Cuda error in func '%s' at line %i : %s.\n", \
		         __FUNCTION__, __LINE__, cudaGetErrorString(err) );   \
		exit(EXIT_FAILURE);                                           \
	}                                                                 \
} while (0)


std::string nvmlErrorToString(nvmlReturn_t result) {
    return std::string(nvmlErrorString(result));
}

void listDevices() {
	try {
    nvmlReturn_t result;
    // Initialise the library.
    result = nvmlInit();
    check(NVML_SUCCESS == result, std::string("Failed to initialise: " + nvmlErrorToString(result) + "\n"));

		string outString;
		int numDevices = getNumDevices();
		for (int i = 0; i < numDevices; ++i) {
			cudaDeviceProp props;
			CUDA_SAFE_CALL(cudaGetDeviceProperties(&props, i));

			outString += to_string(i) + ";\t" + string(props.name) + ";";
			outString += "\tCC" + to_string(props.major) + to_string(props.minor) + ";";
			outString += "\t" + to_string(int(round(props.totalGlobalMem / (1024*1024*1024.0)))) + "GB";

      // Get the device's uuid.
      char pciBusId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
      int arrayLength = NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE;
      CUDA_SAFE_CALL(cudaDeviceGetPCIBusId(pciBusId, arrayLength, i));
      outString += ";\t" + string(pciBusId);

      nvmlDevice_t device;
      result = nvmlDeviceGetHandleByPciBusId(pciBusId, &device);
      if (NVML_SUCCESS != result) {
        cout << "Failed to get handle for device with this PCI bus ID \"" << pciBusId << "\": " << nvmlErrorToString(result) << endl;
        continue;
      }

      char deviceUuid[NVML_DEVICE_UUID_BUFFER_SIZE];
      result = nvmlDeviceGetUUID(device, deviceUuid, NVML_DEVICE_UUID_BUFFER_SIZE);
      if (NVML_SUCCESS == result) {
        outString += ";\t" + string(deviceUuid);
      } else {
        cout << "Failed to get device uuid: " << nvmlErrorToString(result) << endl;
      }
      outString += "\n";
		}
  cout << outString;
	} catch(std::runtime_error const& err) {
		cerr << "CUDA error: " << err.what() << '\n';
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
  cout << "List CUDA devices in CUDA order.\nFORMAT: deviceID;\tdeviceName;\tcompute capability version;\tsize of ram;\tPCI ID;\tUUID\n";
}

int main(int argc, char *argv[]) { 
  if (argc > 1) {
    usage();
  }
  listDevices();
  return 0;
}
