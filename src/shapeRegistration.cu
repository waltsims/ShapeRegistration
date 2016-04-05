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

void centerOfMass (float *imgIn, size_t w, size_t h, float &xCentCoord, float &yCentCoord) {
  int numOfForegroundPixel;

  xCentCoord = 0;
  yCentCoord = 0;
  numOfForegroundPixel = 0;

  for (size_t y = 0; y < h; y++) {
    for (size_t x = 0; x < w; x++) {
      if (imgIn[x + (w * y)] == FOREGROUND) {
          xCentCoord = xCentCoord + x;
          yCentCoord = yCentCoord + y;
          numOfForegroundPixel++;
      }
    }
  }

  xCentCoord /= numOfForegroundPixel;
  yCentCoord /= numOfForegroundPixel;
}

void imgNormalization (float *imgIn, size_t w, size_t h, QuadCoords* qCoords, float xCentCoord, float yCentCoord) {
  /** NOTE: check max(xCentCoord, w - xCentCoord) again */
  float normXFactor = 0.5 / max(xCentCoord, w - xCentCoord);
  float normYFactor = 0.5 / max(yCentCoord, h - yCentCoord);

  size_t index;

  for (size_t y = 0; y < h; y++) {
    for (size_t x = 0; x < w; x++) {
      index = x + (w * y);
      qCoords[index].x[0] = (qCoords[index].x[0] - xCentCoord) * normXFactor;
      qCoords[index].x[1] = (qCoords[index].x[1] - xCentCoord) * normXFactor;
      qCoords[index].x[2] = (qCoords[index].x[2] - xCentCoord) * normXFactor;
      qCoords[index].x[3] = (qCoords[index].x[3] - xCentCoord) * normXFactor;
  
      qCoords[index].y[0] = (qCoords[index].y[0] - yCentCoord) * normYFactor;
      qCoords[index].y[1] = (qCoords[index].y[1] - yCentCoord) * normYFactor;
      qCoords[index].y[2] = (qCoords[index].y[2] - yCentCoord) * normYFactor;
      qCoords[index].y[3] = (qCoords[index].y[3] - yCentCoord) * normYFactor;
    }
  }
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

void tps(float* imgIn, size_t w, size_t h, float *sigma, float *affineParam, float *ctrlP, float *localCoeff, size_t iter) {
  // for (int i = 0 ; i < iter; i++) {
  //   sigma[i] = 0;
  //   for (size_t y = 0; y < h; y++) {
  //     for (size_t x = 0; x < w; x++){
  //       int localP = imgIn[x + (w * y)];
  //       int numP = w * h;, float *sigma, float *sigma, float *sigma
  //       int colInd = x + (w * y);

  //       float radialApproximation = radialApprox(ctrlP, localP, localCoeff, numP, i);
  //       /**   (a_i1 *x_1)  + (a_i2 *x_2) + a_i3
  //        *  = (scale*x_1)  + (sheer*x_2) + translation 
  //        *  = (        rotation        ) + translation
  //        */
  //        sigma[i] = (affineParam[i]*x) + (affineParam[i + 1]*y) + affineParam[i + 2] + radialApproximation;
  //     }
  //   }
  // }
}

float radialApprox(size_t w, size_t h, float *sigma, float *ctrlP, float localP, float *localCoeff, int numP, int iter) {
  // float sigma;
  // float euclidianDist;

  // euclidianDist = 0;
  // for (int j = 0; j < numP; j++) {
  //   euclidianDist = pow((ctrlP[j] - localP), 2) * log(pow((ctrlP[j] - localP), 2));
  //   sigma += localCoeff[i + (colInd*j)] * euclidianDist;
  // }

  // return sigma;
}
