// ###
// ###
// ### Practical Course: GPU Programming in Computer Vision
// ###
// ###
// ### Technical University Munich, Computer Vision Group
// ### Winter Semester 2015/2016, March 15 - April 15
// ###
// ###

#include "testingGPU.h"
#include <iostream>
#include <stdio.h>
#include <cstring>

#define DIM_C_REF 5

using namespace std;

int main(int argc, char **argv) {
  // Before the GPU can process your kernels, a so called "CUDA context" must be
  // initialized
  // This happens on the very first call to a CUDA function, and takes some time
  // (around half a second)
  // We will do it right here, so that the run time measurements are accurate
  cudaDeviceSynchronize();
  CUDA_CHECK;

  // Reading command line parameters:
  // getParam("param", var, argc, argv) looks whether "-param xyz" is specified,
  // and if so stores the value "xyz" in "var"
  // If "-param" is not specified, the value of "var" remains unchanged
  //
  // return value: getParam("param", ...) returns true if "-param" is specified,
  // and false otherwise

  // input template image
  bool ret;
  string templateStr = "";
  ret = getParam("t", templateStr, argc, argv);
  if (!ret) cerr << "ERROR: no template image specified" << endl;

  // input observation image
  string observationStr = "";
  ret = getParam("o", observationStr, argc, argv);
  if (!ret) cerr << "ERROR: no observation image specified" << endl;

  // show the usage instructions
  if (argc <= 3) {
    cout << "Usage: " << argv[0] << " -t <template> -o <observation>" << endl;
    return 1;
  }

  // Load the input image using opencv (load as "grayscale", since we are
  // working only with binary shapes of single channel)
  cv::Mat observationIn =
      cv::imread(observationStr.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
  // check
  if (observationIn.data == NULL) {
    cerr << "ERROR: Could not load observation image " << observationStr
         << endl;
    return 1;
  }

  cv::Mat templateIn = cv::imread(templateStr.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
  // check
  if (templateIn.data == NULL) {
    cerr << "ERROR: Could not load template image " << templateStr << endl;
    return 1;
  }

  // convert to float representation (opencv loads image values as single bytes
  // by default)
  templateIn.convertTo(templateIn, CV_32F);
  // convert range of each channel to [0,1] (opencv default is [0,255])
  templateIn /= 255.f;
  // get image dimensions
  int t_w = templateIn.cols;  // width
  int t_h = templateIn.rows;  // height
  cout << "template image: " << t_w << " x " << t_h << endl;

  observationIn.convertTo(observationIn, CV_32F);
  // convert range of each channel to [0,1] (opencv default is [0,255])
  observationIn /= 255.f;
  // get image dimensions
  int o_w = observationIn.cols;  // width
  int o_h = observationIn.rows;  // height

  cout << "observation image: " << o_w << " x " << o_h << endl;

  // Set the output image format
  cv::Mat mOut(o_h, o_w, CV_32FC1);  // mOut will be a grayscale image, 1 layer
  // ### Define your own output images here as needed

  // Allocate arrays
  // input/output image width: w
  // input/output image height: h

  // allocate raw input image array
  float *observationImg = new float[(size_t)o_w * o_h];
  float *templateImg = new float[(size_t)t_w * t_h];

  // allocate raw output array (the computation result will be stored in this
  // array, then later converted to mOut for displaying)
  float *imgOut = new float[(size_t)o_w * o_h];

  // Init raw input image array
  // opencv images are interleaved: rgb rgb rgb...  (actually bgr bgr bgr...)
  // But for CUDA it's better to work with layered images: rrr... ggg... bbb...
  // So we will convert as necessary, using interleaved "cv::Mat" for
  // loading/saving/displaying, and layered "float*" for CUDA computations

  convert_mat_to_layered(templateImg, templateIn);
  convert_mat_to_layered(observationImg, observationIn);

  Timer timer;
  timer.start();
  testALLGPU(templateImg, templateIn, observationImg, observationIn, t_w, t_h, o_w, o_h, imgOut) ;

  //stop timer here
  timer.end();
  float t = timer.get();  // elapsed time in seconds
  cout << "time: " << t * 1000 << " ms" << endl;

  // wait for key inputs
  cv::waitKey(0);
  // save input and result
  /*cv::imwrite("image_input.png",*/
  /*mIn * 255.f);  // "imwrite" assumes channel range [0,255]*/
  /*cv::imwrite("image_result.png", mOut * 255.f);*/

  // free allocated arrays
/*  delete[] observationImg;
  delete[] templateImg;

  delete[] imgOut;
  delete[] resizedImOut;*/

  // close all opencv windows
  cvDestroyAllWindows();
  return 0;
}
