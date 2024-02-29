/* 
Anisha Kumari Kushwaha
Sarthak Kagliwal
26th January 2024
Image processing filters, effects, and transformations functions
*/

#pragma once

#ifndef FILTER_H
#define FILTER_H

#include <opencv2/core/core.hpp>
#include "faceDetect.h"


int greyscale(cv::Mat &src, cv::Mat &dst);
int applySepiaTone(cv::Mat &src, cv::Mat &dst);
int blur5x5_1(cv::Mat &src, cv::Mat &dst);
int blur5x5_2(cv::Mat &src, cv::Mat &dst);
int sobelX3x3(cv::Mat &src, cv::Mat &dst);
int sobelY3x3(cv::Mat &src, cv::Mat &dst);
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels);
void colorfulFace(cv::Mat &src, cv::Mat &dst, cv::CascadeClassifier &faceCascade);
void embossEffect(cv::Mat &src, cv::Mat &dst);
void blurOutsideFaces(cv::Mat &src, cv::Mat &dst, cv::CascadeClassifier &faceCascade);
void cartoonEffect(cv::Mat &src, cv::Mat &dst);
void adjustBrightness(cv::Mat &src, cv::Mat &dst, int brightness);
void customNegative(cv::Mat &src, cv::Mat &dst);
void invertColorsWithTint(cv::Mat &src, cv::Mat &dst, const cv::Vec3b &tint);
void customColorFilter(cv::Mat &src, cv::Mat &dst, const cv::Vec3f &colorMultiplier);
void artisticBrushStrokeEffect(cv::Mat &src, cv::Mat &dst, int strokeSize);
#endif /* FILTER_H */