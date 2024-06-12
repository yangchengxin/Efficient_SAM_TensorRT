#pragma once

//Choose your device index, default is 0
static const int GpuId = 0;


//the normalization set for preprocess
//bgr
//static const float pixel_mean[] = { 103.53 / 255, 116.28 / 255, 123.675 / 255 };
//static const float pixel_std[] = { 57.375 / 255, 57.12 / 255, 58.395 / 255 };
//rgb
static const float pixel_mean[] = { 123.675 / 255, 116.28 / 255, 103.53 / 255 };
static const float pixel_std[] = { 58.395 / 255, 57.12 / 255, 57.375 / 255 };