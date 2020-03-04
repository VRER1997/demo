//
// Created by c1over on 3/4/20.
//

#ifndef MATMUL_CUUTILS_H
#define MATMUL_CUUTILS_H

class TimerGPU {
public:
    cudaEvent_t start, stop;
    cudaStream_t stream;
    TimerGPU(cudaStream_t stream_ = 0) : stream(stream_) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, stream);
    }
    ~TimerGPU() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    float read() {
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float time;
        cudaEventElapsedTime(&time, start, stop);
        return time;
    }
};

#endif //MATMUL_CUUTILS_H
