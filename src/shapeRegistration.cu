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
	  //bug fix was to switch indexes here.... works for now
	  //TODO look for cause of this and repair in future!!!!
      pCoords[index].x = (float)y;
      pCoords[index].y = (float)x;
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

		float tempPCoordsX = pCoords[index].x;
		float tempPCoordsY = pCoords[index].y;
      // TODO this looks right but seems to be producing incorrect reuslts
      // at the boundaries..... see facebook discussion from sunday
      pCoords[index].x = (tpsParams.affineParam[0] * tempPCoordsX) +
                         (tpsParams.affineParam[1] * tempPCoordsY) +
                         tpsParams.affineParam[2] + freeDeformation[0];

      pCoords[index].y = (tpsParams.affineParam[3] * tempPCoordsX) +
                         (tpsParams.affineParam[4] * tempPCoordsY) +
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
       /*int x = w / 2, y = h / 2;*/

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
            freeDeformation[i] += tpsParams.localCoeff[k + (i * dimSize)] * Q;
          }
        }
/*
        printf("---------------------------------\n");
        printf ("freeDef[0]: %lf, freeDef[1]: %lf\n", freeDeformation[0], freeDeformation[1]);
        printf("---------------------------------\n");
        */


        // note:: change
		float tempQCoordsX = qCoords[index].x[qIndex];
		float tempQCoordsY = qCoords[index].y[qIndex];

        qCoords[index].x[qIndex] =
            (tpsParams.affineParam[0] * tempQCoordsX) +
            (tpsParams.affineParam[1] * tempQCoordsY) +
            tpsParams.affineParam[2] + freeDeformation[0];

        qCoords[index].y[qIndex] =
            (tpsParams.affineParam[3] * tempQCoordsX) +
            (tpsParams.affineParam[4] * tempQCoordsY) +
            tpsParams.affineParam[5] + freeDeformation[1];
	  }
	}
  }
}

float radialApprox( float x, float y, float cx, float cy ) {

  float r2 = (cx - x)*(cx - x) + (cy - y)*(cy - y);

  return r2 < 0.0000000001 ? 0 : r2 * log(r2);

}

void jacobianTrans(int w, int h, float *jacobi, TPSParams &tpsParams,
                   int c_dim) {
  // Index in the image and in the *jacobi
  int indexP;
  // Number of control points
  int K = c_dim * c_dim;
  // Square of the distance of the control point from the pixel
  float squareOfNorm;
  // Common term for all the i,j in each c_k,x combination (precomputed)
  float precomp;
  // x_j (x or y)
  float x_j;
  // Temporary storage of the Jacobian elements, in order to compute the determinant
  float jacEl[4];
  // Index in the local jacEl
  int indexJ;

  // For each pixel
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      // Index of the pixel in the image
      indexP = x + w * y;

      // Reset the local jacobi elements to a_ij for the current pixel
      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
          indexJ = i + 2*j;
          jacEl[indexJ] = tpsParams.affineParam[i + 2*j];
        }
      }

      // Note: synchronize here in the GPU version

      // For each control point
      for (int k = 0; k < K; k++) {
        // Compute the argument of the log()
        // squareOfNorm = (ck_x - x)^2 + (ck_y - y)^2
        squareOfNorm = (tpsParams.ctrlP[k] - x)   * (tpsParams.ctrlP[k] - x)
                     + (tpsParams.ctrlP[k+K] - y) * (tpsParams.ctrlP[k+K] - y);
		//TODO this should be globaly defined as eps
        if (squareOfNorm > 0.000001) {
          // Precompute the reused term
          precomp = 2 * (1 + log(squareOfNorm));
        } else {
          precomp = 2;
        }
        // For each of the four elements of the jacobian
        for (int i = 0; i < 2; i++) {
          for (int j = 0; j < 2; j++) {
            // Index in the local jacobi elements array
            indexJ = i + 2*j;
            // Do we need the x or the y in the place of x_j?
            x_j = (j == 0 ? x : y);
            // jacobi_ij -= precomp * w_ki * (c_kj - x_j)
            jacEl[indexJ] -= precomp * tpsParams.localCoeff[k + i*K]
                            * (tpsParams.ctrlP[k + j*K] - x_j);
          }
        }
      }

      // Compute the determinant of the local jacobi elements
      jacobi[indexP] = jacEl[0]*jacEl[3] - jacEl[1]*jacEl[2];

    }
  }
  return;
}

void transfer(float *imgIn, PixelCoords *pCoords, QuadCoords *qCoords, int t_w,
              int t_h, int o_w, int o_h, float *imgOut) {

  int index;
  int p_index;

  for (int j = 0; j < t_h; j++) {
    for (int i = 0; i < t_w; i++) {
      index = i + t_w * j;
      if (imgIn[index] == FOREGROUND) {
        float xpolygon[4] = {qCoords[index].x[0], qCoords[index].x[1],
                             qCoords[index].x[2], qCoords[index].x[3]};
        float ypolygon[4] = {qCoords[index].y[0], qCoords[index].y[1],
                             qCoords[index].y[2], qCoords[index].y[3]};
        // TODO create local index to search for neignboring points
		// withing bounding box of polygon
        for (int y = 0; y < o_h; y++) {
          for (int x = 0; x < o_w; x++) {
            p_index = x + o_w * y;

			if ( pointInPolygon(4, xpolygon, ypolygon, pCoords[p_index].x,
							  pCoords[p_index].y) )
              imgOut[p_index] = FOREGROUND;
          }
        }
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
      c =! c;
  }

  return c;

}
