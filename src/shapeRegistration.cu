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
#include <stdio.h>

void setQuadCoords (QuadCoords* qCoords, size_t w, size_t h) {
  for (size_t y = 0; y < h; y++) {
    for (size_t x = 0; x < w; x++) {
      qCoords->x[0] = (float)x - 0.5; qCoords->y[0] = (float)y - 0.5;
      qCoords->x[1] = (float)x + 0.5; qCoords->y[1] = (float)y - 0.5;
      qCoords->x[2] = (float)x + 0.5; qCoords->y[2] = (float)y + 0.5;
      qCoords->x[3] = (float)x - 0.5; qCoords->y[3] = (float)y + 0.5;
    }
  }
}

void cutMargins (float* imgIn, size_t w, size_t h, float*& resizedImg, int& resizedW, int& resizedH) {
  int top = -1;
  int bottom = -1;
  int left = -1;
  int right = -1;

  /** set the y-coordinate on the top of the image */
  for (size_t y = 0; y < h; y++) {
    for (size_t x = 0; x < w; x++) {
      if (imgIn[x + (w * y)] == FOREGROUND) {
        top = y;
        break;
      }
    }
    if (top != -1){
      break;
    }
  }

  /** set the y-coordinate on the bottom of the image */
  for (size_t y = h; y > 0; y--) {
    for (size_t x = 0; x < w; x++) {
      if (imgIn[x + (w * y)] == FOREGROUND) {
        bottom = y;
        break;
      }
    }
    if (bottom != -1){
      break;
    }
  }

  /** set the x-coordinate on the left of the image */
  for (size_t x = 0; x < w; x++) {
    for (size_t y = 0; y < h; y++) {
      if (imgIn[x + (w * y)] == FOREGROUND) {
        left = x;
        break;
      }
    }
    if (left != -1){
      break;
    }
  }

  /** set the x-coordinate on the right of the image */
  for (size_t x = w; x > 0; x--) {
    for (size_t y = 0; y < h; y++) {
      if (imgIn[x + (w * y)] == FOREGROUND) {
        right = x;
        break;
      }
    }
    if (right != -1){
      break;
    }
  }

  resizedH = bottom - top + 1;
  resizedW = right - left + 1;

  /** allocate raw input image array */
  resizedImg = new float[resizedW * resizedH];

  for (int y = 0; y < resizedH ; y++) {
    for (int x = 0; x < resizedW ; x++) {
      resizedImg[x + (size_t)(resizedW * y)] = imgIn[(x+left) + (w * (y+top))];
    }
  }
}

void centerOfMass (float *imgIn, size_t w, size_t h, float *xCentCoord, float *yCentCoord) {
  int numOfForegroundPixel;

  xCentCoord[0] = 0;
  yCentCoord[0] = 0;
  numOfForegroundPixel = 0;

  for (size_t y = 0; y < h; y++) {
    for (size_t x = 0; x < w; x++) {
      if (imgIn[x + (w * y)] == FOREGROUND) {
          xCentCoord[0] = xCentCoord[0] + x;
          yCentCoord[0] = yCentCoord[0] + y;
          numOfForegroundPixel++;
      }
    }
  }

  xCentCoord[0] /= numOfForegroundPixel;
  yCentCoord[0] /= numOfForegroundPixel;
}

void imgNormalization (float *imgIn, size_t w, size_t h) {

  //centerOfMass (float *imgIn, size_t w, size_t h, float *xCentCoord, float *yCentCoord);

}

void imageMoment(float *imgIn, size_t w, size_t h, float *mmt, size_t mmtDegree) {
  for (size_t p = 0; p < mmtDegree; p++) {
    for (size_t q = 0; q < mmtDegree; q++) {
      mmt[p + p * q] = 0;

      for (size_t y = 0; y < h; y++) {
        for (size_t x = 0; x < w; x++) {
          /** note: (q+p)th order in the dissertation but not here,
           *  need to check later
           */
          mmt[p + (p * q)] += pow(x, p+1) * pow(y, q+1) * imgIn[x + (w * y)];
        }
      }
    }
  }
}

int pointInPolygon(int nVert, float *vertX, float *vertY, float testX, float testY) {
  /** how we can use???????????????????????????????????????????????*/
  int i, j, c = 0;
  for (i = 0, j = nVert-1; i < nVert; j = i++) {
    if ( ((vertY[i]>testY) != (vertY[j]>testY)) &&
   (testX < (vertX[j]-vertX[i]) * (testY-vertY[i]) / (vertY[j]-vertY[i]) + vertX[i]) )
       c = !c;
  }
  return c;
}

void tps(float *sigma, float *affineParam, float *vectorX, float *ctrlP, float localP, float *localCoeff, int numP, int colInd, size_t w, size_t h) {
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
