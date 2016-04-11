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
  // Index in the image array with sizes w,h.
  int index;

  // Set the pixel coordinates (from (0,0) to (h-1, w-1))
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      index = x + y * w;
      pCoords[index].x = (float)x;
      pCoords[index].y = (float)y;
      // printf("x = %d, y = %d, index = %d, pCoords.x = %f, pCoords.y = %f.\n",
      //       x, y, index, pCoords[index].x, pCoords[index].y);
    }
  }
}

void setQuadCoords(QuadCoords *qCoords, int w, int h) {
  // Index in the image array with sizes w,h.
  int index;

  // Set the coordinates of the four quad points of each pixel
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
// TODO #& is implimented because resizedImg is never initalized
void cutMargins(float *imgIn, int w, int h, float *&resizedImg, int &resizedW,
                int &resizedH, Margins &margins) {
  /** Initialize the the margin positions */
  margins.top = -1;
  margins.bottom = -1;
  margins.left = -1;
  margins.right = -1;

  /** Top: row (y) of the first foreground pixel from top. */
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

  /** Bottom: row (y) of the last foreground pixel from top. */
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

  /** Left: column (x) of the first foreground pixel from left. */
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

  /** Right: column (x) of the last foreground pixel from left. */
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

  /** Height and width of the cropped image */
  resizedH = margins.bottom - margins.top + 1;
  resizedW = margins.right - margins.left + 1;

  /** Allocate the cropped image array */
  resizedImg = new float[resizedW * resizedH];

  /** Assign the respective full image pixels to the cropped image pixels */
  for (int y = 0; y < resizedH; y++) {
    for (int x = 0; x < resizedW; x++) {
      resizedImg[x + resizedW * y] =
          imgIn[(x + margins.left) + (w * (y + margins.top))];
    }
  }
}

void addMargins(float *resizedImg, int resizedW, int resizedH, float *imgOut,
                int w, int h, Margins &margins) {
  /** Assign each resizedImg pixel to the respective imgOut pixel.
   *  x,y: coordinates in the full-size imgOut */
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      // Index in the imgOut array
      int index = x + y * w;
      // Index in the resizedImg array
      int res_index = (x - margins.left) + (y - margins.top) * resizedW;
      // Check if the value is provided in the resizedImg, else set it to
      // background.
      if (x >= margins.left && x <= margins.right && y >= margins.top &&
          y <= margins.bottom) {
        imgOut[index] = resizedImg[res_index];
      } else {
        imgOut[index] = BACKGROUND;
      }
    }
  }
}

void centerOfMass(float *imgIn, int w, int h, float &xCentCoord,
                  float &yCentCoord) {
  /** Initializations
  * xCentCoord, yCentCoord: x,y indices of the center of mass
  * numOfForegroundPixel: number of the foreground pixels */
  xCentCoord = 0;
  yCentCoord = 0;
  int numOfForegroundPixel = 0;

  /** Compute the sum of the coordinates and the number of foreground pixels */
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      if (imgIn[x + (w * y)] == FOREGROUND) {
        xCentCoord += x;
        yCentCoord += y;
        numOfForegroundPixel++;
      }
    }
  }

  /** Average: divide the sum of the coordinates to the number of pixels */
  xCentCoord /= numOfForegroundPixel;
  yCentCoord /= numOfForegroundPixel;
}

// TODO maybe should be called normalize
void pCoordsNormalization(int w, int h, PixelCoords *pCoords, float xCentCoord,
                          float yCentCoord) {
  // Scaling factors per x,y (1/sx, 1/sy in the Matlab implementation)
  float normXFactor = 0.5 / max(xCentCoord, w - xCentCoord);
  float normYFactor = 0.5 / max(yCentCoord, h - yCentCoord);

  // Debug
  printf(
      "normXFactor = %f, normYFactor = %f, xCentCoord = %f, yCentCoord = %f, "
      "width = %d, height = %d\n",
      normXFactor, normYFactor, xCentCoord, yCentCoord, w, h);

  // Index in the image array
  int index;

  /** Normalize the center coordinates of all the pixels:
   *  Shift to make the center of mass the center of the image and
   *  scale to make the image of unit width and height ([-0.5, 0.5])
   */
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      index = x + (w * y);
      pCoords[index].x = (pCoords[index].x - xCentCoord) * normXFactor;
      pCoords[index].y = (pCoords[index].y - yCentCoord) * normYFactor;
    }
  }
}

// TODO maybe should be called normalize
void qCoordsNormalization(int w, int h, QuadCoords *qCoords, float xCentCoord,
                          float yCentCoord) {
  // Scaling factors per x,y (1/sx, 1/sy in the Matlab implementation)
  float normXFactor = 0.5 / max(xCentCoord, w - xCentCoord);
  float normYFactor = 0.5 / max(yCentCoord, h - yCentCoord);

  // Debug
  printf(
      "normXFactor = %f, normYFactor = %f, xCentCoord = %f, yCentCoord = %f, "
      "width = %d, height = %d\n",
      normXFactor, normYFactor, xCentCoord, yCentCoord, w, h);

  // Index in the image array
  int index;

  /** Normalize the quad coordinates of all the pixels:
   *  Shift to make the center of mass the center of the image and
   *  scale to make the image of unit width and height ([-0.5, 0.5])
   */
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      index = x + (w * y);
      qCoords[index].x[0] = (qCoords[index].x[0] - xCentCoord) * normXFactor;
      qCoords[index].y[0] = (qCoords[index].y[0] - yCentCoord) * normYFactor;
      qCoords[index].x[1] = (qCoords[index].x[1] - xCentCoord) * normXFactor;
      qCoords[index].y[1] = (qCoords[index].y[1] - yCentCoord) * normYFactor;
      qCoords[index].x[2] = (qCoords[index].x[2] - xCentCoord) * normXFactor;
      qCoords[index].y[2] = (qCoords[index].y[2] - yCentCoord) * normYFactor;
      qCoords[index].x[3] = (qCoords[index].x[3] - xCentCoord) * normXFactor;
      qCoords[index].y[3] = (qCoords[index].y[3] - yCentCoord) * normYFactor;
    }
  }
}

// TODO maybe should be called invNormalize
void pCoordsDenormalization(int w, int h, PixelCoords *pCoords,
                            float xCentCoord, float yCentCoord) {
  // Scaling factors per x,y (1/sx, 1/sy in the Matlab implementation)
  float normXFactor = 0.5 / max(xCentCoord, w - xCentCoord);
  float normYFactor = 0.5 / max(yCentCoord, h - yCentCoord);

  // Debug
  printf(
      "normXFactor = %f, normYFactor = %f, xCentCoord = %f, yCentCoord = %f, "
      "width = %d, height = %d\n",
      normXFactor, normYFactor, xCentCoord, yCentCoord, w, h);

  // Index in the image array
  int index;

  /** De-normalize the center coordinates of all the pixels:
   *  Shift to restore the center of mass to the original position and
   *  scale to restore the image to the original size
   */
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      index = x + (w * y);
      pCoords[index].x = (pCoords[index].x / normXFactor) + xCentCoord;
      pCoords[index].y = (pCoords[index].y / normYFactor) + yCentCoord;
    }
  }
}

void imageMoment(float *imgIn, int w, int h, float *mmt, int mmtDegree) {
  // Compute all the combinations of the (p+q)-order image moments
  // Keep in mind that p,q go from 0 to mmtDegree-1.
  for (int p = 0; p < mmtDegree; p++) {
    for (int q = 0; q < mmtDegree; q++) {
      // Initialize the current image moment to zero
      mmt[p + (mmtDegree * q)] = 0;

      // Compute the image moments taking the contributions from all the pixels
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
void pTPS(int w, int h, PixelCoords *pCoords, TPSParams &tpsParams, int c_dim) {
  int index;
  int dimSize = c_dim * c_dim;
  float Q;
 float freeDeformation[2] = {0, 0};
  // for every pixel location
  for (int x = 0; x < w; x++) {
    for (int y = 0; y < h; y++) {
      index = x + w * y;

      // for all c_m support coordinates
      freeDeformation[0] = 0;
      freeDeformation[1] = 0;
      for (int k = 0; k < dimSize; k++) {
        // calculate radial approximation
        Q = radialApprox(pCoords[index].x, pCoords[index].y, tpsParams.ctrlP[k],
                         tpsParams.ctrlP[k + dimSize]);

        /**   (a_i1 *x_1)  + (a_i2 *x_2) + a_i3
         *  = (scale*x_1)  + (sheer*x_2) + translation
         *  = (        rotation        ) + translation
         */

        // multiply with weights
        for (int i = 0; i < 2; i++) {
          freeDeformation[i] += tpsParams.localCoeff[k + i * dimSize] * Q;
        }
      }

      // TODO this looks right but seems to be producing incorrect reuslts
      // at the boundaries..... see facebook discussion from sunday
      pCoords[index].x = (tpsParams.affineParam[0] * pCoords[index].x) +
                         (tpsParams.affineParam[1] * pCoords[index].y) +
                         tpsParams.affineParam[2] + freeDeformation[0];

      pCoords[index].y = (tpsParams.affineParam[3] * pCoords[index].x) +
                         (tpsParams.affineParam[4] * pCoords[index].y) +
                         tpsParams.affineParam[5] + freeDeformation[1];
    }
  }
}

void qTPS(int w, int h, QuadCoords *qCoords, TPSParams &tpsParams, int c_dim) {
  int index;
  int dimSize = c_dim * c_dim;
  float Q;
  float freeDeformation[2] = {0, 0};
  for (int x = 0; x < w; x++) {
    for (int y = 0; y < h; y++) {
      index = x + w * y;

      for (int qIndex = 0; qIndex < 4; qIndex++) {
        Q = 0;
         freeDeformation[0] = 0;
         freeDeformation[1] = 0;
        // for all c_m support coordinates
        for (int k = 0; k < dimSize; k++) {
          // calculate radial approximation
          Q = radialApprox(qCoords[index].x[qIndex], qCoords[index].y[qIndex],
                           tpsParams.ctrlP[k], tpsParams.ctrlP[k + dimSize]);

/*          printf ("Q: %lf\n", Q);*/
          /**   (a_i1 *x_1)  + (a_i2 *x_2) + a_i3
           *  = (scale*x_1)  + (sheer*x_2) + translation
           *  = (        rotation        ) + translation
           */

          // multiply with weights
          for (int i = 0; i < 2; i++) {
            freeDeformation[i] += tpsParams.localCoeff[k + i * dimSize] * Q;
          }
        }
/*
        printf("---------------------------------\n");
        printf ("freeDef[0]: %lf, freeDef[1]: %lf\n", freeDeformation[0], freeDeformation[1]);
        printf("---------------------------------\n");
        */

        // note:: change
        qCoords[index].x[qIndex] =
            (tpsParams.affineParam[0] * qCoords[index].x[qIndex]) +
            (tpsParams.affineParam[1] * qCoords[index].y[qIndex]) +
            tpsParams.affineParam[2] + freeDeformation[0];

        qCoords[index].y[qIndex] =
            (tpsParams.affineParam[3] * qCoords[index].x[qIndex]) +
            (tpsParams.affineParam[4] * qCoords[index].y[qIndex]) +
            tpsParams.affineParam[5] + freeDeformation[1];
      }
    }
  }
}

float radialApprox( float x, float y, float cx, float cy ) {

  float r2 = (cx - x)*(cx - x) + (cy - y)*(cy - x);

  return r2 < 0.000001 ? 0 : r2 * log(r2);

}

void jacobianTrans(int w, int h, float *jacobi, TPSParams &tpsParams,
                   float *ctrlP, int c_dim) {
  int index;
  int dimSize = c_dim * c_dim;
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

void transpose(float *imgIn, PixelCoords *pCoords, QuadCoords *qCoords, int w,
               int h, float *imgOut) {
  int index;
  for (int j = 0; j < h; j++) {
    for (int i = 0; i < w; i++) {
      index = i + w * j;
      if (imgIn[index] == FOREGROUND) {
        float xpolygon[4] = {qCoords[index].x[0], qCoords[index].x[1],
                             qCoords[index].x[2], qCoords[index].x[3]};
        float ypolygon[4] = {qCoords[index].y[0], qCoords[index].y[1],
                             qCoords[index].y[2], qCoords[index].y[3]};
        // TODO for foreground points from origional image, if new pixel in
        // polygon --> Pixel = Foreground!
        // TODO create local index to search for neignboring points
        /*for (int j = 0; j < h; j++) {*/
        /*for (int i = 0; i < w; i++) {*/

        if (pointInPolygon(4, xpolygon, ypolygon, pCoords[index].x,
                           pCoords[index].y))
          imgOut[index] = FOREGROUND;
      }
    }
  }
}

bool pointInPolygon(int nVert, float *vertX, float *vertY, float testX,
                    float testY) {
  int i, j;
  bool c = false;
  for (i = 0, j = nVert - 1; i < nVert; j = i++) {
    if (((vertY[i] > testY) != (vertY[j] > testY)) &&
        (testX <
         (vertX[j] - vertX[i]) * (testY - vertY[i]) / (vertY[j] - vertY[i]) +
             vertX[i]))
      c = !c;
  }
  return c;
}
