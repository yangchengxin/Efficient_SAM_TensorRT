#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>


__global__ void print_idx_kernel()
{
    printf("grid dim:(%3d, %3d, %3d), block idx:(%3d, %3d, %3d), thread idx:(%3d, %3d, %3d)\n", gridDim.z, gridDim.y, gridDim.x, blockIdx.z, blockIdx.y, blockIdx.x, threadIdx.z, threadIdx.y, threadIdx.x); 
}

__global__ void print_dim_kernel()
{
    printf("grid dimension:(%3d, %3d, %3d), block dimension:(%3d, %3d, %3d)\n", gridDim.z, gridDim.y, gridDim.x, blockDim.z, blockDim.y, blockDim.x); 
}

__global__ void print_thread_idx_per_block_kernel()
{
    int index = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    printf("block idx: (%3d, %3d, %3d), thread idx: %3d\n", blockIdx.z, blockIdx.y, blockIdx.x, index);
}

__global__ void print_thread_idx_per_grid_kernel()
{
    // 1个block中thread的个数
    int bSize = blockDim.z * blockDim.y * blockDim.x;

    // 线程所在的block在grid中的索引
    int bIndex = blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;

    // thread在某一个block中的索引
    int tIndex = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;

    // bIndex * bsize等于前面所有的thread的个数 加上所在的block中的thread的索引就是该thread在整个grid中的索引
    int index = bIndex * bSize + tIndex;

    printf("block idx: %3d, thread idx in block: %3d, thread idx: %3d\n", bIndex, tIndex, index);
}

__global__ void print_position_kernel()
{
    int index = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    printf("block idx: (%3d, %3d, %3d), thread idx: %3d, cord: (%3d, %3d)\n", blockIdx.z, blockIdx.y, blockIdx.x, index, x, y);
}

void print_one_dim()
{
    int InputSize = 8;
    int blockDim = 4;
    int gridDim = InputSize / blockDim;

    dim3 block(blockDim);
    dim3 grid(gridDim);

    // print_idx_kernel<<<grid, block>>>();
    // print_dim_kernel<<<grid, block>>>();
    // print_thread_idx_per_block_kernel<<<grid, block>>>();
    // print_thread_idx_per_grid_kernel<<<grid, block>>>();
    // print_position_kernel<<<grid, block>>>();

    cudaDeviceSynchronize();
}

void print_two_dim()
{
    int InputWidth = 4;
    int blockDim = 2;
    int gridDim = InputWidth / blockDim;

    dim3 block(blockDim, blockDim);
    dim3 grid(gridDim, gridDim);

    // print_idx_kernel<<<grid, block>>>();
    // print_dim_kernel<<<grid, block>>>();
    // print_thread_idx_per_block_kernel<<<grid, block>>>();
    // print_thread_idx_per_grid_kernel<<<grid, block>>>();
    print_position_kernel<<<grid, block>>>();

    cudaDeviceSynchronize();
}


int main(int argc, char **argv)
{
    /* 笔记
    * 尖括号<<<grid, block>>>， grid表示的是此次会启动的block的个数， block表示的是thread的个数
    * 以print_idx_kernel<<<2, 3>>>()为例，这里我们指定了block的个数是2，thread的个数为3，因此grid的维度为(112)，2表示两个block，每个block有3个线程
    */
    // std::cout << "----- " << "prind_idx_kernel" << " -----" << std::endl; 
    // print_idx_kernel<<<2, 3>>>();
    // cudaDeviceSynchronize();

    // std::cout << "----- " << "prind_idx_kernel" << " -----" << std::endl; 
    // print_dim_kernel<<<2, 3>>>();
    // cudaDeviceSynchronize();

    // std::cout << "----- " << "print_thread_idx_per_block_kernel" << " -----" << std::endl;
    // print_thread_idx_per_block_kernel<<<5, 8>>>();
    // cudaDeviceSynchronize();

    // std::cout << "----- " << "print_thread_idx_per_grid_kernel" << " -----" << std::endl;
    // print_thread_idx_per_grid_kernel<<<2, 3>>>();
    // cudaDeviceSynchronize();

    // std::cout << "----- " << "print_position_kernel" << " -----" << std::endl;
    // print_position_kernel<<<2, 8>>>();
    // cudaDeviceSynchronize();

    // print_one_dim();

    print_two_dim();

    return 0;
}