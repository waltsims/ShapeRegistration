/**
 *  \brief     Shape Registration code
 *  \details   Functions for Nonlinear Shape Registration without
 * Correspondences
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

void setPixelCoords(PixelCoords *pCoords, int w, int h) {
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      int index = x + y * w;
      pCoords[index].x = (float)x;
      pCoords[index].y = (float)y;
    }
  }
}

void setQuadCoords(QuadCoords *qCoords, int w, int h) {
  int index;

  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      index = x + y * w;
      qCoords[index].x[0] = (float)x - 0.5;
      qCoords[index].y[0] = (float)y - 0.5;
      qCoords[index].x[1] = (float)x + 0.5;
      qCoords[index].y[1] = (float)y - 0.5;
      qCoords[index].x[2] = (float)x + 0.5;
      qCoords[index].y[2] = (float)y + 0.5;
      qCoords[index].x[3] = (float)x - 0.5;
      qCoords[index].y[3] = (float)y + 0.5;
    }
  }
}

// TODO maybe should be called crop
void cutMargins(float *imgIn, int w, int h, float *&resizedImg, int &resizedW,
                int &resizedH, Margins &margins) {
  margins.top = -1;
  margins.bottom = -1;
  margins.left = -1;
  margins.right = -1;

  /** set the y-coordinate on the top of the image */
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      if (imgIn[x + (w * y)] == FOREGROUND) {
        margins.top = y;
        break;
      }
    }
    if (margins.top != -1) {
      break;
    }
  }

  /** set the y-coordinate on the bottom of the image */
  for (int y = h - 1; y >= 0; y--) {
    for (int x = 0; x < w; x++) {
      if (imgIn[x + (w * y)] == FOREGROUND) {
        margins.bottom = y;
        break;
      }
    }
    if (margins.bottom != -1) {
      break;
    }
  }

  /** set the x-coordinate on the left of the image */
  for (int x = 0; x < w; x++) {
    for (int y = 0; y < h; y++) {
      if (imgIn[x + (w * y)] == FOREGROUND) {
        margins.left = x;
        break;
      }
    }
    if (margins.left != -1) {
      break;
    }
  }

  /** set the x-coordinate on the right of the image */
  for (int x = w - 1; x >= 0; x--) {
    for (int y = 0; y < h; y++) {
      if (imgIn[x + (w * y)] == FOREGROUND) {
        margins.right = x;
        break;
      }
    }
    if (margins.right != -1) {
      break;
    }
  }

  resizedH = margins.bottom - margins.top + 1;
  resizedW = margins.right - margins.left + 1;

  /** allocate raw input image array */
  resizedImg = new float[resizedW * resizedH];

  for (int y = 0; y < resizedH; y++) {
    for (int x = 0; x < resizedW; x++) {
      resizedImg[x + (int)(resizedW * y)] =
          imgIn[(x + margins.left) + (w * (y + margins.top))];
    }
  }
}

void addMargins(float *resizedImg, int resizedW, int resizedH, float *imgOut,
                int w, int h, Margins &margins) {
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      // TODO if value outside boundaries then BACKGROUND else set as resisze
      // image
      int index = x + y * w;
      int rel_index = (x - margins.left) + (y - margins.top) * resizedW;
      if (x >= margins.left && x <= margins.right && y >= margins.top &&
          y <= margins.bottom) {
        imgOut[index] = resizedImg[rel_index];
      }
    }
  }
}

void centerOfMass(float *imgIn, int w, int h, float &xCentCoord,
                  float &yCentCoord) {
  int numOfForegroundPixel;

  xCentCoord = 0;
  yCentCoord = 0;
  numOfForegroundPixel = 0;

  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
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

// TODO maybe should be called normalize
void pCoordsNormalization(int w, int h, PixelCoords *pCoords, float xCentCoord,
                          float yCentCoord) {
  /** NOTE: check max(xCentCoord, w - xCentCoord) again */
  float normXFactor = 0.5 / max(xCentCoord, w - xCentCoord);
  float normYFactor = 0.5 / max(yCentCoord, h - yCentCoord);

  int index;

  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      index = x + (w * y);
      pCoords[index].x = (pCoords[index].x - xCentCoord) * normXFactor;
      pCoords[index].y = (pCoords[index].y + yCentCoord) * normYFactor;
    }
  }
}

// TODO maybe should be called normalize
void qCoordsNormalization(int w, int h, QuadCoords *qCoords, float xCentCoord,
                          float yCentCoord) {
  /** NOTE: check max(xCentCoord, w - xCentCoord) again */
  float normXFactor = 0.5 / max(xCentCoord, w - xCentCoord);
  float normYFactor = 0.5 / max(yCentCoord, h - yCentCoord);

  int index;

  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      index = x + (w * y);
      qCoords[index].x[0] = (qCoords[index].x[0] - xCentCoord) * normXFactor;
      qCoords[index].y[0] = (qCoords[index].y[0] - xCentCoord) * normYFactor;
      qCoords[index].x[1] = (qCoords[index].x[1] - xCentCoord) * normXFactor;
      qCoords[index].y[1] = (qCoords[index].y[1] - xCentCoord) * normYFactor;
      qCoords[index].x[2] = (qCoords[index].x[2] - xCentCoord) * normXFactor;
      qCoords[index].y[2] = (qCoords[index].y[2] - xCentCoord) * normYFactor;
      qCoords[index].x[3] = (qCoords[index].x[3] - xCentCoord) * normXFactor;
      qCoords[index].y[3] = (qCoords[index].y[3] - xCentCoord) * normYFactor;
    }
  }
}

// TODO maybe should be called invNormalize
void imgDenormalization(int w, int h, PixelCoords *pCoords, float xCentCoord,
                        float yCentCoord) {
  /** NOTE: check max(xCentCoord, w - xCentCoord) again */
  float normXFactor = 0.5 / max(xCentCoord, w - xCentCoord);
  float normYFactor = 0.5 / max(yCentCoord, h - yCentCoord);

  int index;

  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      index = x + (w * y);
      pCoords[index].x = (pCoords[index].x / normXFactor) + xCentCoord;
      pCoords[index].y = (pCoords[index].y / normYFactor) - yCentCoord;
    }
  }
}

void imageMoment(float *imgIn, int w, int h, float *mmt, int mmtDegree) {
  for (int p = 0; p < mmtDegree; p++) {
    for (int q = 0; q < mmtDegree; q++) {
      mmt[p + (mmtDegree * q)] = 0;

      for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
          /** note: (q+p)th order in the dissertation but not here,
           *  need to check later
           */
          mmt[p + (mmtDegree * q)] +=
              pow(x, p + 1) * pow(y, q + 1) * imgIn[x + (w * y)];
        }
      }
    }
  }
}

// PixelCoords* pCoordsSigma
void pTPS(int w, int h, PixelCoords* pCoords, TPSParams &tpsParams, int mmtDegree) {
  int index;
  for (int x = 0; x < w; x++) {
    for (int y = 0; y < h; y++) {
      index = x + w * y;

      float* radialApproximation = pTPSradialApprox(tpsParams, pCoords[index], mmtDegree);

      /**   (a_i1 *x_1)  + (a_i2 *x_2) + a_i3
       *  = (scale*x_1)  + (sheer*x_2) + translation
       *  = (        rotation        ) + translation
       */
      pCoords[index].x = (tpsParams.affineParam[0] * pCoords[index].x) +
          (tpsParams.affineParam[1] * pCoords[index].y) +
          tpsParams.affineParam[2] + radialApproximation[0];

      pCoords[index].y = (tpsParams.affineParam[3] * pCoords[index].y) +
          (tpsParams.affineParam[4] * pCoords[index].y) +
          tpsParams.affineParam[5] + radialApproximation[1];
    }
  }
}

float* pTPSradialApprox(TPSParams &tpsParams, PixelCoords pCoords, int mmtDegree) {

  float euclidianDist[2] = {0, 0};
  float* sum = new float[2];
  int index;  

  int dimSize = mmtDegree * mmtDegree;


  for (int i = 0; i < dimSize; i++) {
    for (int j = 0; j < 2; j++) {
      index = i + j * dimSize;
      // NOTE: change power function to a by a
      float r = sqrt(pow((tpsParams.ctrlP[index] - pCoords.x), 2) +
               pow((tpsParams.ctrlP[index] - pCoords.y), 2));

      euclidianDist[j] = pow(r, 2) * log(pow(r, 2));
      sum[j] += tpsParams.localCoeff[index] * euclidianDist[j];

    }
  }

  return sum;
}

void qTPS(int w, int h, QuadCoords *qCoords, TPSParams &tpsParams,
          int mmtDegree) {
  int index;
  for (int x = 0; x < w; x++) {
    for (int y = 0; y < h; y++) {
      for (int qIndex = 0; qIndex < 4; qIndex++) {
        index = x + w * y;

        float *radialApproximation =
            qTPSradialApprox(tpsParams, qCoords[index], mmtDegree, qIndex);

        /**   (a_i1 *x_1)  + (a_i2 *x_2) + a_i3
         *  = (scale*x_1)  + (sheer*x_2) + translation
         *  = (        rotation        ) + translation
         */

        // note:: change
        qCoords[index].x[qIndex] =
            (tpsParams.affineParam[0] * qCoords[index].x[qIndex]) +
            (tpsParams.affineParam[1] * qCoords[index].y[qIndex]) +
            tpsParams.affineParam[2] + radialApproximation[0];

        qCoords[index].y[qIndex] =
            (tpsParams.affineParam[3] * qCoords[index].y[qIndex]) +
            (tpsParams.affineParam[4] * qCoords[index].y[qIndex]) +
            tpsParams.affineParam[5] + radialApproximation[1];
      }
    }
  }
}

float *qTPSradialApprox(TPSParams &tpsParams, QuadCoords qCoords, int mmtDegree,
                        int qIndex) {
  float euclidianDist[2] = {0, 0};
  float *sum = new float[2];
  int index;

  int dimSize = mmtDegree * mmtDegree;


  for (int i = 0; i < dimSize; i++) {
    for (int j = 0; j < 2; j++) {
      // NOTE: change power function to a by a
      float r = sqrt(pow((tpsParams.ctrlP[index] - qCoords.x[qIndex]), 2) +
                     pow((tpsParams.ctrlP[index] - qCoords.y[qIndex]), 2));
      
      euclidianDist[j] = pow(r, 2) * log(pow(r, 2));
      sum[j] += tpsParams.localCoeff[index] * euclidianDist[j];
      index = i + j * dimSize;
    }
  }
  return sum;
}

void jacobianTrans(int w, int h, float *jacobi, TPSParams &tpsParams,
                   float *ctrlP, int mmtDegree) {
  int index;
  int dimSize = mmtDegree * mmtDegree;
  float radialApprox;
  float squareOfNorm;

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      index = i + 2 * j;
      jacobi[index] = tpsParams.affineParam[index];

      radialApprox = 0;
      for (int x = 0; x < w; x++) {
        for (int y = 0; y < h; y++) {
          squareOfNorm = 0;
          for (int k = 0; k < dimSize; i++) {
            int indexNorm = k + dimSize * i;
            squareOfNorm = pow((tpsParams.ctrlP[indexNorm] - x), 2) +
                           pow((tpsParams.ctrlP[indexNorm] - y), 2);
          }
          radialApprox += 2 *
                          tpsParams.localCoeff[x + w * h + (i * (x + w * h))] *
                          (ctrlP[x + w * h + (j * (x + w * h))] - y) *
                          (1 + log(squareOfNorm));
        }
      }
    }
  }
}

int pointInPolygon(int nVert, float *vertX, float *vertY, float testX,
                   float testY) {
  /** how we can use???????????????????????????????????????????????*/
  int i, j, c = 0;
  for (i = 0, j = nVert - 1; i < nVert; j = i++) {
    if (((vertY[i] > testY) != (vertY[j] > testY)) &&
        (testX <
         (vertX[j] - vertX[i]) * (testY - vertY[i]) / (vertY[j] - vertY[i]) +
             vertX[i]))
      c = !c;
  }
  return c;
}

