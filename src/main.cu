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
#include <iostream>
using namespace std;

/** calculate moment of the image
 *  \param[in] imgIn         input image
 *  \param[in] w             size of width of the image
 *  \param[in] h             size of height of the image
 *  \param[in] nc            the number of channels of the image
 *  \param[out] mmt          an array for moments of the image
 *  \param[out] mmtDegree    the degree of moments
 *
 *  \retrun nothing
 *  \note pseudo code of geometric
 *moments(http://de.mathworks.com/matlabcentral/answers/71678-how-to-write-matlab-code-for-moments)
 */
void imageMoment(float *imgIn, size_t w, size_t h, size_t nc, float *mmt,
                 size_t mmtDegree) {
  for (int p = 0; p < mmtDegree; p++) {
    for (int q = 0; q < mmtDegree; p++) {
      mmt[p + p * q] = 0;

      for (int c = 0; c < nc; c++) {
        for (int y = 0; y < h; y++) {
          for (int x = 0; x < w; x++) {
            mmt[p + (p * q)] +=
                pow(x, p) * pow(y, q) * imgIn[x + (w * y) + (w * h * c)];
          }
        }
      }
    }
  }
}


/** thin plate spline
 *  \param[in]  affineParam        affine parameters
 *  \param[in]  vectorX            vector X
 *  \param[in]  localCoeff         local coefficients
 *  \param[in] numP          index k
 *  \param[in] colInd        index i
 *
 *  \retrun nothing
 *  \note https://en.wikipedia.org/wiki/Thin_plate_spline
 */
void tps(float *sigma, float *affineParam, float *vectorX, float *localCoeff, float *interP, int numP, int colInd){

  for (int i = 0 ; i < colInd; i++) {
    sigma[i] = 0;
    sigma[i] = (a[i]*vectorX[1]) + (a[i + 1]*vectorX[2]) + (a[i + 2]*vectorX[3]) + radialApprox(interP, centP, localCoeff);
  }
}

/** radial basis approximation
 *  \param[in] interP        c_k
 *  \param[in] centP         x
 *  \param[in] localCoeff    w_ki
 *  \param[in] numP          index k
 *  \param[in] colInd        index i
 *  \param[out] sigma        to be added
 *
 *  \retrun give a new location
 *  \note https://en.wikipedia.org/wiki/Radial_basis_function
 */
float radialApprox(float *interP, float centP, float *localCoeff, int numP, int colInd){
  float sigma;
  float euclidianDist;

  euclidianDist = 0;
  for (int j = 0; j < numP; j++) {
    euclidianDist = pow((interP[j] - centP), 2) * log(pow((interP[j] - centP), 2));
    sigma += localcoeff[i + (colInd*j)] * euclidianDist; 
  }
  
  return sigma;
}

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
    cout << "Usage: " << argv[0] << " -i <image> [-repeats <repeats>] [-gray]"
         << endl;
    return 1;
  }

  // number of computation repetitions to get a better run time measurement
  int repeats = 1;
  getParam("repeats", repeats, argc, argv);
  cout << "repeats: " << repeats << endl;

  // load the input image as grayscale if "-gray" is specifed
  bool gray = false;
  getParam("gray", gray, argc, argv);
  cout << "gray: " << gray << endl;

  // ### Define your own parameters here as needed

  // Load the input image using opencv (load as grayscale if "gray==true",
  // otherwise as is (may be color or grayscale))
  cv::Mat mIn =
      cv::imread(image.c_str(), (gray ? CV_LOAD_IMAGE_GRAYSCALE : -1));
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
  int nc = mIn.channels();  // number of channels
  cout << "image: " << w << " x " << h << endl;

  // Set the output image format
  // ###
  // ###
  // ### TODO: Change the output image format as needed
  // ###
  // ###
  cv::Mat mOut(h, w, mIn.type());  // mOut will have the same number of channels
                                   // as the input image, nc layers
  // cv::Mat mOut(h,w,CV_32FC3);    // mOut will be a color image, 3 layers
  // cv::Mat mOut(h,w,CV_32FC1);    // mOut will be a grayscale image, 1 layer
  // ### Define your own output images here as needed

  // Allocate arrays
  // input/output image width: w
  // input/output image height: h
  // input image number of channels: nc
  // output image number of channels: mOut.channels(), as defined above (nc, 3,
  // or 1)

  // allocate raw input image array
  float *imgIn = new float[(size_t)w * h * nc];

  // allocate raw output array (the computation result will be stored in this
  // array, then later converted to mOut for displaying)
  float *imgOut = new float[(size_t)w * h * mOut.channels()];

  // Init raw input image array
  // opencv images are interleaved: rgb rgb rgb...  (actually bgr bgr bgr...)
  // But for CUDA it's better to work with layered images: rrr... ggg... bbb...
  // So we will convert as necessary, using interleaved "cv::Mat" for
  // loading/saving/displaying, and layered "float*" for CUDA computations
  convert_mat_to_layered(imgIn, mIn);

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
  convert_layered_to_mat(mOut, imgOut);
  showImage("Output", mOut, 100 + w + 40, 100);

  // ### Display your own output images here as needed

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
