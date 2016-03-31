/** 
 *  \brief     Shape Registration code
 *  \details   Functions for Nonlinear Shape Registration without Correspondences
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

#include "shapeRegistration.h"

void imageMoment(float *imgIn, size_t w, size_t h, size_t nc, float *mmt,
                 size_t mmtDegree) {
  for (size_t p = 0; p < mmtDegree; p++) {
    for (size_t q = 0; q < mmtDegree; q++) {
      mmt[p + p * q] = 0;

      for (size_t c = 0; c < nc; c++) {
        for (size_t y = 0; y < h; y++) {
          for (size_t x = 0; x < w; x++) {
            mmt[p + (p * q)] +=
                pow(x, p) * pow(y, q) * imgIn[x + (w * y) + (w * h * c)];
          }
        }
      }
    }
  }
}

int pointInPolygon(int nVert, float *vertX, float *vertY, float testX, float testY) {
  int i, j, c = 0;
  for (i = 0, j = nVert-1; i < nVert; j = i++) {
    if ( ((vertY[i]>testY) != (vertY[j]>testY)) &&
   (testX < (vertX[j]-vertX[i]) * (testY-vertY[i]) / (vertY[j]-vertY[i]) + vertX[i]) )
       c = !c;
  }
  return c;
}

void tps(float *sigma, float *affineParam, float *vectorX, float *ctrlP, float localP, float *localCoeff, int numP, int colInd, size_t w, size_t h, size_t nc) {
  for (size_t c = 0; c < nc; c++) {
    for (size_t y = 0; y < h; y++) {
      for (size_t x = 0; x < w; x++){
          for (int i = 0 ; i < colInd; i++) {
            sigma[i] = 0;
            float radialApproximation = radialApprox(ctrlP, localP, localCoeff, numP, i, colInd);
            sigma[i] = (affineParam[i]*vectorX[1]) + (affineParam[i + 1]*vectorX[2]) + (affineParam[i + 2]*vectorX[3]) + radialApproximation;
          }
      }
    }
  }

}

float radialApprox(float *ctrlP, float localP, float *localCoeff, int numP, int i, int colInd) {
  float sigma;
  float euclidianDist;

  euclidianDist = 0;
  for (int j = 0; j < numP; j++) {
    euclidianDist = pow((ctrlP[j] - localP), 2) * log(pow((ctrlP[j] - localP), 2));
    sigma += localCoeff[i + (colInd*j)] * euclidianDist; 
  }
  
  return sigma;
}