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
#include "shapeRegistrationGPU.h"
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

//testALLGPU(templateImg, templateIn, observationImg, observationIn, t_w, t_h, o_w, o_h, imgOut);

void testALLGPU(float *templateImg, cv::Mat templateIn, float *observationImg, cv::Mat observationIn, int t_w, int t_h, int o_w, int o_h, float *imgOut){

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
  printf("xCentTemplate = %f, yCentTemplate = %f\n", xCentTemplate, yCentTemplate);
  QuadCoords *qTemplate = new QuadCoords[rt_w * rt_h];
  setQuadCoordsGPU(qTemplate, rt_w, rt_h);
  float t_sx = 1, t_sy = 1; // Normalisation factors

  qCoordsNormalisationGPU(rt_w, rt_h, qTemplate, xCentTemplate, yCentTemplate, t_sx, t_sy);
  PixelCoords *pTemplate = new PixelCoords[rt_w * rt_h];
  setPixelCoordsGPU(pTemplate, rt_w, rt_h);
  pCoordsNormalisationGPU(rt_w, rt_h, pTemplate, xCentTemplate, yCentTemplate, t_sx, t_sy);
  printf("t_sx = %f, t_sy = %f\n", t_sx, t_sy);
  printf("detN1 = %f\n", t_sx*t_sy);

  // TPS transformation parameters
  TPSParams tpsParams;

  // normalized pCoords of the Observation
  float xCentObservation;
  float yCentObservation;
  centerOfMassGPU(resizedObservation, ro_w, ro_h, xCentObservation, yCentObservation);
  printf("xCentObservation = %f, yCentObservation = %f\n", xCentObservation, yCentObservation);
  PixelCoords *pResizedObservation = new PixelCoords[ro_w * ro_h];
  setPixelCoordsGPU(pResizedObservation, ro_w, ro_h);
  float o_sx = 1, o_sy = 1; // Normalisation factors
  pCoordsNormalisationGPU(ro_w, ro_h, pResizedObservation, xCentObservation, yCentObservation, o_sx, o_sy);
  printf("o_sx = %f, o_sy = %f\n", o_sx, o_sy);
  printf("detN2 = %f\n", o_sx*o_sy);


  /*transfer(resizedTemplate, pResizedObservation, qTemplate, rt_w, rt_h, ro_w, ro_h,*/
           /*resizedImOut);*/

  /**printf("[Main] tpsParams affine first: %f, last: %f\n", tpsParams.affineParam[0], tpsParams.affineParam[5]);
  printf("[Main] tpsParams localC first: %f, last: %f\n", tpsParams.localCoeff[0], tpsParams.localCoeff[2 * DIM_C_REF * DIM_C_REF - 1]);
  printf("[Main] rt_w = %d, rt_h = %d, ro_w = %d, ro_h = %d\n", rt_w, rt_h, ro_w, ro_h);
  printf("[Main] templateImg first = %f, last = %f\n", resizedTemplate[0], resizedTemplate[rt_w * rt_h - 1]);
  printf("[Main] observationImg first = %f, last = %f\n", resizedObservation[0], resizedObservation[rt_w * rt_h - 1]);
  printf("[Main] Normalisation first = %f, last = %f\n", normFactor[0], normFactor[80]);
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
  // 81,              Normalisation
  // 2 * rt_w * rt_h, pTemplate
  // 8 * rt_w * rt_h, qTemplate
  // 2 * rt_w * rt_h, pObservation
  // 1,               t_sx,
  // 1,               t_sy,
  // 1,               o_sx
  // 1,               o_sy

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
  // Normalisation factors (N_i for eq.22)
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
  int m_dat = 87;
  // Parameter collection for tuning the fit procedure.
  lm_control_struct control = lm_control_float;
  // Verbosity level
  // control.verbosity = 10;
  // Relative error desired in the sum of squares.
  control.ftol = 0.0001;
  // Relative error between last two approximations.
  control.xtol = 0.0001;
  // max function evaluations = patience*n_par
  control.patience = 10;
  // Progress messages will be written to this file. (NULL --> stdout)
  control.msgfile = NULL;
  // Status object
  lm_status_struct status;

  // Call the lmmin() using the wrapper for the objective function
  lmmin( sizePar, par, m_dat, data, lmminObjectiveWrapperGPU, &control, &status );

  // Translate the found vector of parameters to the tpsParams
  // Unack the affineParam
  printf("TPSParam:\n");
  for (int i = 0; i < 6; i++) {
    tpsParams.affineParam[i] = par[i];
    printf("%f\n", tpsParams.affineParam[i]); // Debug
  }
  // Unpack the localCoeff
  for (int i = 0; i < 2 * DIM_C_REF * DIM_C_REF; i++) {
    tpsParams.localCoeff[i] = par[i+6];
    printf("%f\n", tpsParams.localCoeff[i]); // Debug
  }

  // compensating for the translation caused by image cropping (see Matlab)
  float t_tx = -(xCentTemplate    + templateMargins.top    ) * t_sx;
  float t_ty = -(yCentTemplate    + templateMargins.left   ) * t_sy;
  float o_tx = -(xCentObservation + observationMargins.top ) * o_sx;
  float o_ty = -(yCentObservation + observationMargins.left) * o_sy;
  // Debug
  printf("t_sx = %f, t_sy = %f, t_tx = %f, t_ty = %f", t_sx, t_sy, t_tx, t_ty);
  printf("o_sx = %f, o_sy = %f, o_tx = %f, o_ty = %f", o_sx, o_sy, o_tx, o_ty);

  // Denormalize the coefficients for the final transformation
  for (int j = 0; j < 3; j++) {
    tpsParams.affineParam[j] /= o_sx;
    tpsParams.affineParam[3+j] /= o_sy;
  }
  printf("affineParam[2] = %f, o_tx = %f, o_sx = %f, division = %f\n", tpsParams.affineParam[2], o_tx, o_sx, o_tx / o_sx); // Debug
  tpsParams.affineParam[2] -= o_tx / o_sx;
  tpsParams.affineParam[5] -= o_ty / o_sy;

  for (int j = 0; j < DIM_C_REF * DIM_C_REF; j++) {
    tpsParams.localCoeff[j] /= o_sx;
    tpsParams.localCoeff[DIM_C_REF * DIM_C_REF + j] /= o_sy;
  }

  // Debug
  printf("Denormalized TPSParam:\n");
  for (int i = 0; i < 6; i++) {
    printf("%f\n", tpsParams.affineParam[i]);
  }
  for (int i = 0; i < 2 * DIM_C_REF * DIM_C_REF; i++) {
    printf("%f\n", tpsParams.localCoeff[i]);
  }
  //

  // Apply the decided transformation on the normalized quad coordinates of the template
  qTPSGPU(rt_w, rt_h, qTemplate, tpsParams, DIM_C_REF);

  // Restore the pResizedObservation to the full size
  // void pCoordsDenormalisation(int w, int h, PixelCoords *pCoords,
                              // float xCentCoord, float yCentCoord) {
  pCoordsDenormalisationGPU(ro_w, ro_h, pResizedObservation, xCentObservation, yCentObservation);
printf("pResizedObservation: [0] = %f, %f, [last] = %f, %f\n", pResizedObservation[0].x, pResizedObservation[0].y, pResizedObservation[ro_w * ro_h-1].x, pResizedObservation[ro_w * ro_h-1].y);
  // Transfer (map) the transformed quad coordinates to pixel coordinates.
  // Store the result to the resizedImOut
  transferGPU(resizedTemplate, pResizedObservation, qTemplate, rt_w, rt_h, ro_w, ro_h, resizedImOut);

  // Debug
  printf("pResizedObservation: [0] = %f, %f, [last] = %f, %f\n", pResizedObservation[0].x, pResizedObservation[0].y, pResizedObservation[ro_w * ro_h-1].x, pResizedObservation[ro_w * ro_h-1].y);
  printf("qTemplate[0].x[0]: %f, [0].y[0]: %f, [last].x[0]: %f, [last].y[0]: %f\n", qTemplate[0].x[0], qTemplate[0].y[0], qTemplate[rt_w * rt_h-1].x[0], qTemplate[rt_w * rt_h-1].y[0]);
  //

  // Convert and show the transformed output
  convert_layered_to_mat(resizedImgOut, resizedImOut);
  showImage("Resized Output", resizedImgOut, 800, 100);
}
