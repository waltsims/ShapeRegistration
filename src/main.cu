// ###
// ###
// ### Practical Course: GPU Programming in Computer Vision
// ###
// ###
// ### Technical University Munich, Computer Vision Group
// ### Winter Semester 2015/2016, March 15 - April 15
// ###
// ###

#include "helper.h"
#include "shapeRegistration.h"
#include <iostream>
#include <stdio.h>
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

  // input image
  string image = "";
  bool ret = getParam("i", image, argc, argv);
  if (!ret) cerr << "ERROR: no image specified" << endl;
  if (argc <= 1) {
    cout << "Usage: " << argv[0] << " -i <image>"
         << endl;
    return 1;
  }

  // Load the input image using opencv (load as "grayscale", since we are working
  // only with binary shapes of single channel)
  cv::Mat mIn = cv::imread(image.c_str(), CV_LOAD_IMAGE_GRAYSCALE );
  // check
  if (mIn.data == NULL) {
    cerr << "ERROR: Could not load image " << image << endl;
    return 1;
  }

  // convert to float representation (opencv loads image values as single bytes
  // by default)
  mIn.convertTo(mIn, CV_32F);
  // convert range of each channel to [0,1] (opencv default is [0,255])
  mIn /= 255.f;
  // get image dimensions
  int w = mIn.cols;         // width
  int h = mIn.rows;         // height
  cout << "image: " << w << " x " << h << endl;

  // Set the output image format
  cv::Mat mOut(h,w,CV_32FC1);    // mOut will be a grayscale image, 1 layer
  // ### Define your own output images here as needed

  // Allocate arrays
  // input/output image width: w
  // input/output image height: h

  // allocate raw input image array
  float *imgIn = new float[(size_t)w * h];

  // allocate raw output array (the computation result will be stored in this
  // array, then later converted to mOut for displaying)
  float *imgOut = new float[(size_t)w * h];

  // Init raw input image array
  // opencv images are interleaved: rgb rgb rgb...  (actually bgr bgr bgr...)
  // But for CUDA it's better to work with layered images: rrr... ggg... bbb...
  // So we will convert as necessary, using interleaved "cv::Mat" for
  // loading/saving/displaying, and layered "float*" for CUDA computations
  convert_mat_to_layered(imgIn, mIn); // Replace this to remove the conversions, we don't use channels.

  Timer timer;
  timer.start();
  // ###
  // ###
  // ### TODO: Main computation
  // ###
  // ###
  timer.end();
  float t = timer.get();  // elapsed time in seconds
  cout << "time: " << t * 1000 << " ms" << endl;

  // show input image
  showImage("Input", mIn, 100,
            100);  // show at position (x_from_left=100,y_from_above=100)

  // show output image: first convert to interleaved opencv format from the
  // layered raw array
  convert_layered_to_mat(mOut, imgOut); // Replace this to remove the conversions, we don't use channels.
  showImage("Output", mOut, 100 + w + 40, 100);

  // ### Display your own output images here as needed

  /** function testings are from here */

  float *resizedImg;
  int resizedW;
  int resizedH;


  cutMargins (imgIn, w, h, resizedImg, resizedW, resizedH);

  cv::Mat resizedImgOut(resizedH, resizedW, CV_32FC1);
  convert_layered_to_mat(resizedImgOut, resizedImg);
  showImage("Resized Output", resizedImgOut, 100 + w + 40 + w + 40, 100);

  printf("%d, %d\n", resizedW, resizedH);

  QuadCoords* qCoords = new QuadCoords[resizedH * resizedW];
  setQuadCoords (qCoords, resizedW, resizedH);



  /** function testings are to here */

  // wait for key inputs
  cv::waitKey(0);

  // save input and result
  cv::imwrite("image_input.png",
              mIn * 255.f);  // "imwrite" assumes channel range [0,255]
  cv::imwrite("image_result.png", mOut * 255.f);

  // free allocated arrays
  delete[] imgIn;
  delete[] imgOut;

  // close all opencv windows
  cvDestroyAllWindows();
  return 0;
}
