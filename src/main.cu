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
#include "lmmin.h"
#include "testing.h"
#include <iostream>
#include <stdio.h>
#include <cstring>

#define DIM_C_REF 5

double  normFactor[81] = {1.5707963267949,
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

  // show input images
  showImage("templateIn", templateIn, 100,
            100);  // show at position (x_from_left=100,y_from_above=100)
  showImage("observatinIn", observationIn, 300,
            100);  // show at position (x_from_left=100,y_from_above=100)

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

  cv::Mat resizedImgOut(ro_h, ro_w, CV_32FC1);  // mOut will be a grayscale image, 1 layer
  float *resizedImOut = new float[ro_w * ro_h];
  convert_layered_to_mat(resizedImgOut, resizedObservation);
  showImage("observation with cut margins", resizedImgOut, 550, 100);

  // we also need the center of mass for normailisation
  float xCentTemplate;
  float yCentTemplate;

  //normalized quadCoords of Template
  centerOfMass(resizedTemplate, rt_w, rt_h, xCentTemplate, yCentTemplate);
	printf("xCentTemplate = %f, yCentTemplate = %f\n", xCentTemplate, yCentTemplate);
  QuadCoords *qTemplate = new QuadCoords[rt_w * rt_h];
  setQuadCoords(qTemplate, rt_w, rt_h);
	float t_sx = 1, t_sy = 1; // Normalisation factors
  qCoordsNormalization(rt_w, rt_h, qTemplate, xCentTemplate, yCentTemplate, t_sx, t_sy);
  PixelCoords *pTemplate = new PixelCoords[rt_w * rt_h];
  setPixelCoords(pTemplate, rt_w, rt_h);
  pCoordsNormalisation(rt_w, rt_h, pTemplate, xCentTemplate, yCentTemplate, t_sx, t_sy);
	printf("t_sx = %f, t_sy = %f\n", t_sx, t_sy);
	printf("detN1 = %f\n", t_sx*t_sy);

  // TPS transformation parameters
  TPSParams tpsParams;

	// normalized pCoords of the Observation
	float xCentObservation;
	float yCentObservation;
	centerOfMass(resizedObservation, ro_w, ro_h, xCentObservation, yCentObservation);
	printf("xCentObservation = %f, yCentObservation = %f\n", xCentObservation, yCentObservation);
  PixelCoords *pResizedObservation = new PixelCoords[ro_w * ro_h];
  setPixelCoords(pResizedObservation, ro_w, ro_h);
	float o_sx = 1, o_sy = 1; // Normalisation factors
	pCoordsNormalisation(ro_w, ro_h, pResizedObservation, xCentObservation, yCentObservation, o_sx, o_sy);
	printf("o_sx = %f, o_sy = %f\n", o_sx, o_sy);
	printf("detN2 = %f\n", o_sx*o_sy);


  /*transfer(resizedTemplate, pResizedObservation, qTemplate, rt_w, rt_h, ro_w, ro_h,*/
           /*resizedImOut);*/


  /*convert_layered_to_mat(resizedImgOut, resizedImOut);*/
  /*showImage("Resized Output", resizedImgOut, 800, 100);*/

	/**printf("[Main] tpsParams affine first: %f, last: %f\n", tpsParams.affineParam[0], tpsParams.affineParam[5]);
	printf("[Main] tpsParams localC first: %f, last: %f\n", tpsParams.localCoeff[0], tpsParams.localCoeff[2 * DIM_C_REF * DIM_C_REF - 1]);
	printf("[Main] rt_w = %d, rt_h = %d, ro_w = %d, ro_h = %d\n", rt_w, rt_h, ro_w, ro_h);
	printf("[Main] templateImg first = %f, last = %f\n", resizedTemplate[0], resizedTemplate[rt_w * rt_h - 1]);
	printf("[Main] observationImg first = %f, last = %f\n", resizedObservation[0], resizedObservation[rt_w * rt_h - 1]);
	printf("[Main] normalization first = %f, last = %f\n", normFactor[0], normFactor[80]);
	printf("[Main] pTemplate first.x = %f, first.y = %f, last.x = %f, last.y = %f\n", pTemplate[0].x, pTemplate[0].y, pTemplate[rt_w * rt_h-1].x, pTemplate[rt_w * rt_h-1].y);
	printf("[Main] qTemplate first.x[0] = %f, first.y[3] = %f, last.x[0] = %f, last.y[3] = %f\n", qTemplate[0].x[0], qTemplate[0].y[3], qTemplate[rt_w * rt_h-1].x[0], qTemplate[rt_w * rt_h-1].y[3]);
	printf("[Main] pObservation first.x = %f, first.y = %f, last.x = %f, last.y = %f\n", pResizedObservation[0].x, pResizedObservation[0].y, pResizedObservation[ro_w * ro_h -1].x, pResizedObservation[ro_w * ro_h -1].y);
*/


	// Pack the parameters for the lmmin() objective function
	int sizePar = 6 + (2 * DIM_C_REF * DIM_C_REF);
	double *par = new double[sizePar];
	// Pack the affineParam
	for (int i = 0; i < 6; i++) {
		par[i] = tpsParams.affineParam[i];
	}
	// Pack the localCoeff
	for (int i = 0; i < 2 * DIM_C_REF * DIM_C_REF; i++) {
		par[i+6] = tpsParams.localCoeff[i];
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

	int sizeData = (4) + (rt_w * rt_h) + (ro_w * ro_h) + (81) + (2 * rt_w * rt_h)
							 + (8 * rt_w * rt_h) + (2 * ro_w * ro_h) + (4);
	float *data = new float[sizeData];
	// current writing position in the data array
	int offset = 0;

	// Pack the sizes of the arrays
	data[offset    ] = rt_w;
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
		data[offset + 2*i]   = pTemplate[i].x;
		data[offset + 2*i+1] = pTemplate[i].y;
	}
	offset += 2 * rt_w * rt_h;
	// Quad coordinates of the template
	// Every element has two fields (x,y) that are arrays of four elements (corners)
	for (int i = 0; i < rt_w * rt_h; i++) {
		data[offset + 8*i  ] = qTemplate[i].x[0];
		data[offset + 8*i+1] = qTemplate[i].y[0];
		data[offset + 8*i+2] = qTemplate[i].x[1];
		data[offset + 8*i+3] = qTemplate[i].y[1];
		data[offset + 8*i+4] = qTemplate[i].x[2];
		data[offset + 8*i+5] = qTemplate[i].y[2];
		data[offset + 8*i+6] = qTemplate[i].x[3];
		data[offset + 8*i+7] = qTemplate[i].y[3];
	}
	offset += 8 * rt_w * rt_h;
	// Pixel coordinates of the observation
  // Every element is a struct with two fields: x, y
  for (int i = 0; i < ro_w * ro_h; i++) {
    data[offset + 2*i]   = pResizedObservation[i].x;
    data[offset + 2*i+1] = pResizedObservation[i].y;
  }
  offset += 2 * ro_w * ro_h;
	// Normalisation factors of the template
	data[offset    ] = t_sx;
	data[offset + 1] = t_sy;
	offset += 2;
	// Normalisation factors of the observation
	data[offset    ] = o_sx;
	data[offset + 1] = o_sy;
	offset += 2;

	// Configuration parameters for the lmmin()
	// Number of equations
	int m_dat = 87; // TODO: add the 6 extra equations
	lm_control_struct control = lm_control_float;
	lm_status_struct status;

	// Call the lmmin() using the wrapper for the objective function
	lmmin( sizePar, par, m_dat, data, lmminObjectiveWrapper, &control, &status );

	// double residual[87] = {};
	// objectiveFunction(resizedObservation, resizedTemplate, ro_w, ro_h,
  //                   normFactor, tpsParams, qTemplate, pTemplate,
  //                   pResizedObservation, rt_w, rt_h, residual);


	// objectiveFunction(observationImg, templateImg, ro_w, ro_h,
  //                   normalization, tpsParams, qTemplate, pTemplate,
  //                   pObservation, rt_w, rt_h, residual);


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
  delete[] observationImg;
  delete[] templateImg;

  delete[] imgOut;
  delete[] resizedImOut;

  // close all opencv windows
  cvDestroyAllWindows();
  return 0;
}
