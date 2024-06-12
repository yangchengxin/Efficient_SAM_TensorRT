#pragma once
// #ifndef CPPDLL_EXPORTS
// #define CPPDLL_API __declspec(dllexport)
// #else
// #define CPPDLL_API __declspec(dllimport)
// #endif

#include <iostream>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <NvInferRuntime.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <math.h>
#include <fstream>
#include <vector>
#include <memory>
#include <functional>
#include <opencv2/opencv.hpp>
#include <assert.h>

using namespace std;
using namespace cv;
using namespace nvinfer1;