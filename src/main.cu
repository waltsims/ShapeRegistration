// ###
// ###
// ### Practical Course: GPU Programming in Computer Vision
// ###
// ###
// ### Technical University Munich, Computer Vision Group
// ### Winter Semester 2015/2016, March 15 - April 15
// ###
// ###

#include <stdio.h>
#include <cstring>
#include <iostream>
#include "helper.h"
#include "lmmin.h"
#include "shapeRegistration.h"
#include "shapeRegistrationGPU.h"
// #include "testingGPU.h"

#define DIM_C_REF 5

double normFactor[81] = {1.5707963267949,
                         0.471404520791033,
                         0.196349540849362,
                         0.0942809041582067,
                         0.0490873852123405,
                         0.026937401188059,
                         0.0153398078788564,
                         0.00897913372935302,
                         0.00536893275759974,
                         0.471404520791033,
                         0.125,
                         0.0471404520791033,
                         0.0208333333333333,
                         0.0101015254455221,
                         0.00520833333333333,
                         0.00280597929042281,
                         0.0015625,
                         0.00089281159240726,
                         0.196349540849362,
                         0.0471404520791033,
                         0.0163624617374468,
                         0.00673435029701476,
                         0.00306796157577128,
                         0.00149652228822551,
                         0.00076699039394282,
                         0.000408142442243319,
                         0.000223705531566656,
                         0.0942809041582067,
                         0.0208333333333333,
                         0.00673435029701476,
                         0.00260416666666667,
                         0.00112239171616913,
                         0.000520833333333333,
                         0.000255089026402074,
                         0.000130208333333333,
                         0.0000686778148005585,
                         0.0490873852123405,
                         0.0101015254455221,
                         0.00306796157577128,
                         0.00112239171616913,
                         0.000460194236365692,
                         0.000204071221121659,
                         0.0000958737992428525,
                         0.000047093358720383,
                         0.0000239684498107131,
                         0.0269374011880590,
                         0.00520833333333333,
                         0.00149652228822551,
                         0.000520833333333333,
                         0.000204071221121659,
                         0.0000868055555555556,
                         0.0000392444656003192,
                         0.0000186011904761905,
                         0.0000091570419734078,
                         0.0153398078788564,
                         0.00280597929042281,
                         0.00076699039394282,
                         0.000255089026402074,
                         0.0000958737992428525,
                         0.0000392444656003192,
                         0.0000171203212933665,
                         0.00000784889312006383,
                         0.00000374507028292392,
                         0.00897913372935302,
                         0.0015625,
                         0.000408142442243319,
                         0.000130208333333333,
                         0.0000470933587203830,
                         0.0000186011904761905,
                         0.00000784889312006383,
                         0.00000348772321428571,
                         0.00000161594858354255,
                         0.00536893275759974,
                         0.00089281159240726,
                         0.000223705531566656,
                         0.0000686778148005585,
                         0.0000239684498107131,
                         0.0000091570419734078,
                         0.00000374507028292392,
                         0.00000161594858354255,
                         0.000000728208110568542};

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

  // maximum number of iterations for the Levenberg-Marquardt (patience)
  int patience = 25;
  getParam("l", patience, argc, argv);
  if (patience < 1) {
    cerr << "ERROR: the patience for the Levenberg-Marquardt must be >=1"
         << endl;
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

  // allocate raw input image array
  float *observationImg = new float[(size_t)o_w * o_h];
  float *templateImg = new float[(size_t)t_w * t_h];

  // Init raw input image array
  // opencv images are interleaved: rgb rgb rgb...  (actually bgr bgr bgr...)
  // But for CUDA it's better to work with layered images: rrr... ggg... bbb...
  // So we will convert as necessary, using interleaved "cv::Mat" for
  // loading/saving/displaying, and layered "float*" for CUDA computations

  convert_mat_to_layered(templateImg, templateIn);
  convert_mat_to_layered(observationImg, observationIn);

  Timer timer;
  timer.start();

  // testALLGPU(templateImg, templateIn, observationImg, observationIn, t_w,
  // t_h, o_w, o_h, imgOut) ;

  // show input images
  showImage("templateIn", templateIn, 100, 100);
  showImage("observationIn", observationIn, 310, 100);

  float *resizedTemplate;
  float *resizedObservation;
  int rt_w;  // resized template width
  int rt_h;  // resized template height
  int ro_w;  // resized observation width
  int ro_h;  // resized observation height

  Margins observationMargins;
  Margins templateMargins;

  cutMargins(templateImg, t_w, t_h, resizedTemplate, rt_w, rt_h,
             templateMargins);
  cutMargins(observationImg, o_w, o_h, resizedObservation, ro_w, ro_h,
             observationMargins);

  // we also need the center of mass for normailisation
  float xCentTemplate;
  float yCentTemplate;

  // normalized quadCoords of Template
  centerOfMass(resizedTemplate, rt_w, rt_h, xCentTemplate, yCentTemplate);

  QuadCoords *qTemplate = new QuadCoords[rt_w * rt_h];
  setQuadCoords(qTemplate, rt_w, rt_h);
  float t_sx = 1, t_sy = 1;  // Normalisation factors
  qCoordsNormalization(rt_w, rt_h, qTemplate, xCentTemplate, yCentTemplate,
                       t_sx, t_sy);
  PixelCoords *pTemplate = new PixelCoords[rt_w * rt_h];
  setPixelCoords(pTemplate, rt_w, rt_h);
  pCoordsNormalisation(rt_w, rt_h, pTemplate, xCentTemplate, yCentTemplate,
                       t_sx, t_sy);

  // TPS transformation parameters
  TPSParams tpsParams;

  float xCentObservation;
  float yCentObservation;
  centerOfMass(resizedObservation, ro_w, ro_h, xCentObservation,
               yCentObservation);

  PixelCoords *pResizedObservation = new PixelCoords[ro_w * ro_h];
  setPixelCoords(pResizedObservation, ro_w, ro_h);
  float o_sx = 1, o_sy = 1;  // Normalisation factors
  pCoordsNormalisation(ro_w, ro_h, pResizedObservation, xCentObservation,
                       yCentObservation, o_sx, o_sy);

  // Pack the parameters for the lmmin() objective function
  int sizePar = 6 + (2 * DIM_C_REF * DIM_C_REF);
  double *par = new double[sizePar];
  // Pack the affineParam
  for (int i = 0; i < 6; i++) {
    par[i] = tpsParams.affineParam[i];
  }
  // Pack the localCoeff
  for (int i = 0; i < 2 * DIM_C_REF * DIM_C_REF; i++) {
    par[i + 6] = tpsParams.localCoeff[i];
  }

  // Pack the auxiliary data for the lmmin() objective function
  // Data format (all floats) [number of elements, name]:
  // 1,               rt_w
  // 1,               rt_h
  // 1,               ro_w
  // 1,               ro_h
  // rt_w * rt_h,     templateImg
  // ro_w * ro_h,     observationImg
  // 81,              normalization
  // 2 * rt_w * rt_h, pTemplate
  // 8 * rt_w * rt_h, qTemplate
  // 2 * rt_w * rt_h, pObservation
  // 1,								t_sx,
  // 1,								t_sy,
  // 1,								o_sx
  // 1,								o_sy

  int sizeData = (4) + (rt_w * rt_h) + (ro_w * ro_h) + (81) +
                 (2 * rt_w * rt_h) + (8 * rt_w * rt_h) + (2 * ro_w * ro_h) +
                 (4);
  float *data = new float[sizeData];
  // current writing position in the data array
  int offset = 0;

  // Pack the sizes of the arrays
  data[offset] = rt_w;
  data[offset + 1] = rt_h;
  data[offset + 2] = ro_w;
  data[offset + 3] = ro_h;
  // We wrote 4 elements, move the reading position 4 places
  offset += 4;
  // Template image array
  for (int i = 0; i < rt_w * rt_h; i++) {
    data[offset + i] = resizedTemplate[i];
  }
  offset += rt_w * rt_h;
  // Observation image array
  for (int i = 0; i < ro_w * ro_h; i++) {
    data[offset + i] = resizedObservation[i];
  }
  offset += ro_w * ro_h;
  // Normalization factors (N_i for eq.22)
  for (int i = 0; i < 81; i++) {
    data[offset + i] = normFactor[i];
  }
  offset += 81;
  // Pixel coordinates of the template
  // Every element is a struct with two fields: x, y
  for (int i = 0; i < rt_w * rt_h; i++) {
    data[offset + 2 * i] = pTemplate[i].x;
    data[offset + 2 * i + 1] = pTemplate[i].y;
  }
  offset += 2 * rt_w * rt_h;
  // Quad coordinates of the template
  // Every element has two fields (x,y) that are arrays of four elements
  // (corners)
  for (int i = 0; i < rt_w * rt_h; i++) {
    data[offset + 8 * i] = qTemplate[i].x[0];
    data[offset + 8 * i + 1] = qTemplate[i].y[0];
    data[offset + 8 * i + 2] = qTemplate[i].x[1];
    data[offset + 8 * i + 3] = qTemplate[i].y[1];
    data[offset + 8 * i + 4] = qTemplate[i].x[2];
    data[offset + 8 * i + 5] = qTemplate[i].y[2];
    data[offset + 8 * i + 6] = qTemplate[i].x[3];
    data[offset + 8 * i + 7] = qTemplate[i].y[3];
  }
  offset += 8 * rt_w * rt_h;
  // Pixel coordinates of the observation
  // Every element is a struct with two fields: x, y
  for (int i = 0; i < ro_w * ro_h; i++) {
    data[offset + 2 * i] = pResizedObservation[i].x;
    data[offset + 2 * i + 1] = pResizedObservation[i].y;
  }
  offset += 2 * ro_w * ro_h;
  // Normalisation factors of the template
  data[offset] = t_sx;
  data[offset + 1] = t_sy;
  offset += 2;
  // Normalisation factors of the observation
  data[offset] = o_sx;
  data[offset + 1] = o_sy;
  offset += 2;

  // Configuration parameters for the lmmin()
  // Number of equations
  int m_dat = 87;
  // Parameter collection for tuning the fit procedure.
  lm_control_struct control = lm_control_float;
  // Verbosity level
  control.verbosity = 1;
  // Relative error desired in the sum of squares.
  control.ftol = 0.0001;
  // Relative error between last two approximations.
  control.xtol = 0.0001;
  // max function evaluations = patience*n_par
  control.patience = patience;
  printf("Solver contol.patience: %d (%d objective function calls)\n",
         control.patience, control.patience * 56);

  // Progress messages will be written to this file. (NULL --> stdout)
  control.msgfile = NULL;
  // Status object
  lm_status_struct status;

  // Call the lmmin() using the wrapper for the objective function
  printf("\nSolving the system...\n");
  lmmin(sizePar, par, m_dat, data, lmminObjectiveWrapper, &control, &status);
  // lmmin(sizePar, par, m_dat, data, lmminObjectiveWrapperGPU, &control,
  // &status);
  printf("Solving completed!\n\n");

  // Translate the found vector of parameters to the tpsParams
  // Unpack the affineParam
  for (int i = 0; i < 6; i++) {
    tpsParams.affineParam[i] = par[i];
  }
  // Unpack the localCoeff
  for (int i = 0; i < 2 * DIM_C_REF * DIM_C_REF; i++) {
    tpsParams.localCoeff[i] = par[i + 6];
  }

  // compensating for the translation caused by image cropping (see Matlab)
  float o_tx = -(xCentObservation + observationMargins.top) * o_sx;
  float o_ty = -(yCentObservation + observationMargins.left) * o_sy;

  // Denormalize the coefficients for the final transformation
  for (int j = 0; j < 3; j++) {
    tpsParams.affineParam[j] /= o_sx;
    tpsParams.affineParam[3 + j] /= o_sy;
  }
  tpsParams.affineParam[2] -= o_tx / o_sx;
  tpsParams.affineParam[5] -= o_ty / o_sy;

  for (int j = 0; j < DIM_C_REF * DIM_C_REF; j++) {
    tpsParams.localCoeff[j] /= o_sx;
    tpsParams.localCoeff[DIM_C_REF * DIM_C_REF + j] /= o_sy;
  }

  // Apply the decided transformation on the normalized quad coordinates of the
  // template
  qTPS(rt_w, rt_h, qTemplate, tpsParams, DIM_C_REF);

  // Find the dimensions needed to fit the registered shape
  int x_min = 0, x_max = 0, y_min = 0, y_max = 0;
  for (int i = 0; i < rt_w * rt_h; i++) {
    for (int q = 0; q < 4; q++) {
      if (qTemplate[i].x[q] < x_min) x_min = qTemplate[i].x[q];
      if (qTemplate[i].x[q] > x_max) x_max = qTemplate[i].x[q];
      if (qTemplate[i].y[q] < y_min) y_min = qTemplate[i].y[q];
      if (qTemplate[i].y[q] > y_max) y_max = qTemplate[i].y[q];
    }
  }

  // Dimensions of the full registered shape image
  int reg_w = x_max - x_min + 1;
  int reg_h = y_max - y_min + 1;
  float *registered = new float[reg_w * reg_h];

  // TODO: The transfer function requires the output to be pre-initialized.
  // Change the implementation of the transfer() and remove this.
  for (int i = 0; i < reg_w * reg_h; i++) {
    registered[i] = 0;
  }

  // Transfer (map) the transformed quad coordinates to pixel coordinates.
  // Store the result to the pRegistered
  PixelCoords *pRegistered = new PixelCoords[reg_w * reg_h];
  setPixelCoords(pRegistered, reg_w, reg_h);
  transfer(registered, pRegistered, reg_w, reg_h, resizedTemplate, qTemplate,
           rt_w, rt_h);

  // Crop the result
  Margins registeredMargins;
  float *resizedRegistered;
  int rreg_w, rreg_h;
  cutMargins(registered, reg_w, reg_h, resizedRegistered, rreg_w, rreg_h,
             registeredMargins);

  // Convert and show the transformed output
  cv::Mat resizedImRegistered(rreg_h, rreg_w, CV_32FC1);
  convert_layered_to_mat(resizedImRegistered, resizedRegistered);
  showImage("Registered shape", resizedImRegistered, 520, 100);

  // stop timer here
  timer.end();
  float t = timer.get();  // elapsed time in seconds
  cout << "time: " << t * 1000 << " ms" << endl;
  // wait for key inputs
  cv::waitKey(0);

  // save input and result
  cv::imwrite("image_template.png",
              templateIn * 255.f);  // "imwrite" assumes channel range [0,255]*/
  printf("Template image was written in the image_template.png.\n");
  cv::imwrite("image_observation.png", observationIn * 255.f);
  printf("Observation image was written in the image_observation.png.\n");
  cv::imwrite("image_registered.png", resizedImRegistered * 255.f);
  printf("Registered shape image was written in the image_registered.png.\n");

  // free allocated arrays
  delete[] observationImg;
  delete[] templateImg;
  delete[] registered;
  delete[] resizedRegistered;
  delete[] qTemplate;
  delete[] pTemplate;
  delete[] pResizedObservation;
  delete[] pRegistered;
  delete[] par;
  delete[] data;

  // close all opencv windows
  cvDestroyAllWindows();
  return 0;
}
