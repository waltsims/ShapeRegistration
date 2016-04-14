/**
 *  \brief     Tests for each function
 *  \details   Unit tests
 *  \author    Gerasimos Chourdakis
 *  \author    Sungjae Jung
 *  \author    Walter Simson
 *  \version   1.0cse
 *  \date      Mar. 2016
 *  \pre       to be added
 *  \bug       to be added
 *  \warning   to be added
 *  \copyright to be added
 */

#include "testingGPU.h"
#include "shapeRegistration.h"
#define DIM_C_REF 5

//testALLGPU(templateImg, templateIn, observationImg, observationIn, t_w, t_h, o_w, o_h, imgOut);

void testImageMomentGPU(float *imgIn, PixelCoords *pImg, int w, int h, float * mmt, int mmtDegree ){

	float * temp = new float[w * h * mmtDegree *mmtDegree];

	imageMomentGPU( imgIn, pImg, w, h , temp, mmtDegree);
	imageMoment( imgIn, pImg, w, h , mmt, mmtDegree);

	float diff = 0.0;
	for (int index = 0; index < w * h * mmtDegree * mmtDegree; index ++){
		diff =+ temp[index] - mmt[index];
	
	}

	if (diff == 0){
		cout << "GPU mmt diff successful" << endl;
	}
	else{
		cout << "mmt diff function has a total discrepincy of: " << diff << endl;
	}

	delete[] temp;


}

void testALLGPU(float *templateImg, cv::Mat templateIn, float *observationImg, cv::Mat observationIn, int t_w, int t_h, int o_w, int o_h, float *imgOut) {
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

  cutMarginsGPU(templateImg, t_w, t_h, resizedTemplate, rt_w, rt_h,
             templateMargins);
  cutMarginsGPU(observationImg, o_w, o_h, resizedObservation, ro_w, ro_h,
             observationMargins);

  cv::Mat resizedImgOut(ro_h, ro_w, CV_32FC1);  // mOut will be a grayscale image, 1 layer
  float *resizedImOut = new float[ro_w * ro_h];
  convert_layered_to_mat(resizedImgOut, resizedObservation);
  showImage("observation with cut margins", resizedImgOut, 550, 100);

  // we also need the center of mass for normailisation
  float xCentTemplate;
  float yCentTemplate;

  //normalized quadCoords of Template
  centerOfMassGPU(resizedTemplate, rt_w, rt_h, xCentTemplate, yCentTemplate);
  QuadCoords *qTemplate = new QuadCoords[rt_w * rt_h];
  setQuadCoordsGPU(qTemplate, rt_w, rt_h);
  qCoordsNormalizationGPU(rt_w, rt_h, qTemplate, xCentTemplate, yCentTemplate);
  PixelCoords *pResizedTemplate = new PixelCoords[rt_w * rt_h];
  setPixelCoordsGPU(pResizedTemplate, rt_w, rt_h);
  pCoordsNormalizationGPU(rt_w, rt_h, pResizedTemplate, xCentTemplate, yCentTemplate);

  // Time to transform the template
  TPSParams tpsParams;

  qTPSGPU(rt_w, rt_h, qTemplate, tpsParams, DIM_C_REF);

  PixelCoords *pResizedObservation = new PixelCoords[ro_w * ro_h];
  setPixelCoordsGPU(pResizedObservation, ro_w, ro_h);

  transferGPU(resizedTemplate, pResizedObservation, qTemplate, rt_w, rt_h, ro_w, ro_h,
           resizedImOut);


  convert_layered_to_mat(resizedImgOut, resizedImOut);
  showImage("Resized Output", resizedImgOut, 800, 100);
	
 
  float * mmt = new float[rt_w * rt_h * 9 * 9 ];
  testImageMomentGPU ( resizedTemplate, pResizedTemplate, rt_w, rt_h , mmt, 9);
  delete[] mmt;

}
