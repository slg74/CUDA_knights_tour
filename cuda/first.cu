#include <stdio.h>

__global__ void helloFromGPU(void) {
    printf("Hello from GPU!\n"); 
}

int main(void) {
    printf("Hello from GPU!\n"); 

    helloFromGPU <<<1, 10>>>();
    cudaDeviceReset();
    return 0;
}
