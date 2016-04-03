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

struct quadCoords {
  float x[4]; 
  float y[4];
};

void setQuadCoords (quadCoords* qCoords, size_t w, size_t h, size_t nc, int xCoord, int yCoord) {
  for (size_t c = 0; c < nc; c++) {
    for (size_t y = 0; y < h; y++) {
      for (size_t x = 0; x < w; x++) {
        qCoords->x[0] = xCoord - 0.5; qCoords->y[0] = yCoord - 0.5;
        qCoords->x[1] = xCoord + 0.5; qCoords->y[1] = yCoord - 0.5;
        qCoords->x[2] = xCoord + 0.5; qCoords->y[2] = yCoord + 0.5;        
        qCoords->x[3] = xCoord - 0.5; qCoords->y[3] = yCoord + 0.5;
      } 
    }
  }
}

float* cutMargins (float* imgIn, size_t w, size_t h, size_t nc, int& resizedW, int& resizedH) {
  int top = -1;
  int bottom = -1;
  int left = -1;
  int right = -1;
  float* resizedImg;

  /** set the y-coordinate on the top of the image */
  for (size_t c = 0; c < nc; c++) {
    for (size_t y = 0; y < h; y++) {
      for (size_t x = 0; x < w; x++) {
        if (imgIn[x + (w * y) + (w * h * c)] == FOREGROUND) {
          top = y;
          printf("top: %d\n", top);
          break;
        }
      }
      if (top != -1){
        break;
      }
    }
    if (top != -1){
      break;
    }
  }

  /** set the y-coordinate on the bottom of the image */
  for (size_t c = 0; c < nc; c++) {
    for (size_t y = h; y > 0; y--) {
      for (size_t x = 0; x < w; x++) {
        if (imgIn[x + (w * y) + (w * h * c)] == FOREGROUND) {
          bottom = y;
          printf("bottom: %d\n", bottom);
          break;
        }
      }
      if (bottom != -1){
        break;
      }
    }
    if (bottom != -1){
      break;
    }
  }

  /** just for testing, print all the valuses in imgIn array */
  // for (size_t c = 0; c < nc; c++) {
  //   for (size_t y = 0; y < h; y++) {
  //     printf("\n==============================================%zu\n", y);
  //     for (size_t x = 0; x < w; x++) {
  //       printf("%.1f ", imgIn[x + (w * y) + (w * h * c)]);
  //     }
  //   }
  // }

  /** set the x-coordinate on the left of the image */
  for (size_t c = 0; c < nc; c++) {
    for (size_t x = 0; x < w; x++) {
      for (size_t y = 0; y < h; y++) {
        if (imgIn[x + (w * y) + (w * h * c)] == FOREGROUND) {
          left = x;
          printf("left: %d\n", left);
          break;
        }
      }
      if (left != -1){
        break;
      }
    }
    if (left != -1){
      break;
    }
  }

  /** set the x-coordinate on the right of the image */
  for (size_t c = 0; c < nc; c++) {
    for (size_t x = w; x > 0; x--) {
      for (size_t y = 0; y < h; y++) {
        if (imgIn[x + (w * y) + (w * h * c)] == FOREGROUND) {
          right = x;
          printf("right: %d\n", right);
          break;
        }
      }
      if (right != -1){
        break;
      }
    }
    if (right != -1){
      break;
    }
  }

  resizedH = bottom - top + 1;
  resizedW = right - left + 1;
  printf("nc: %zu\n", nc );
  printf("resizedH: %d\n", resizedH);
  printf("resizedW: %d\n", resizedW);

  // allocate raw input image array

  resizedImg = new float[resizedW * resizedH * nc];
  // TODO: make resized image and return the image

  /** just for testing, print all the valuses in imgIn array */  
  // for (size_t c = 0; c < nc; c++) {
  //   for (size_t x = 0; x < w; x++) {
  //     printf("\n==============================================%zu\n", x);
  //     for (size_t y = 0; y < h; y++) {
  //       printf("%.1f ", imgIn[x + (w * y) + (w * h * c)]);
  //     }
  //   }
  // }

  // for (size_t c = 0; c < nc; c++) {
  //   for (size_t x = 0; x < w; x++) {
  //     printf("\n==============================================%zu\n", x);
  //     for (size_t y = 0; y < h; y++) {
  //       resizedImg[x + (w * y) + (w * h * c)] = 0;
  //       printf("%.1f ", resizedImg[x + (w * y) + (w * h * c)]);
  //     }
  //   }
  // }



  for (size_t c = 0; c < nc; c++) {
    for (int y = 0; y < resizedH ; y++) {
      for (int x = 0; x < resizedW ; x++) {
        printf("%zu | ", x + (resizedW * y) + (resizedW * resizedH * c));//printf("%.1f ", imgIn[x + (w * y) + (w * h * c)]);
        resizedImg[x + (size_t)(resizedW * y) + (size_t)(resizedW * resizedH * c)] = 0;
        //resizedImg[(nc-1)*(resizedH-1)*(resizedW-1)] = 0;
        //resizedImg[x + (w * y) + (w * h * c)] = imgIn[x + (w * y) + (w * h * c)];
        // resizedImg[x + (w * y) + (w * h * c)] = imgIn[(x+left) + (w * (y+top)) + (w * h * c)];
      }
    }
  }

  return resizedImg;


}

void centerOfMass (float *imgIn, size_t w, size_t h, size_t nc, float *xCentCoord, float *yCentCoord) {
  int numOfForegroundPixel; 

  for (size_t c = 0; c < nc; c++) {
    xCentCoord[c] = 0;
    yCentCoord[c] = 0;
    numOfForegroundPixel = 0; 

    for (size_t y = 0; y < h; y++) {
      for (size_t x = 0; x < w; x++) {
        if (imgIn[x + (w * y) + (w * h * c)] == FOREGROUND)
            xCentCoord[c] = xCentCoord[c] + x;
            yCentCoord[c] = yCentCoord[c] + y;
            numOfForegroundPixel++;
      } 
    }

    xCentCoord[c] /= numOfForegroundPixel;
    yCentCoord[c] /= numOfForegroundPixel;
  }
}

void imgNormalization (float *imgIn, size_t w, size_t h, size_t nc) {

  //centerOfMass (float *imgIn, size_t w, size_t h, size_t nc, float *xCentCoord, float *yCentCoord);

}

void imageMoment(float *imgIn, size_t w, size_t h, size_t nc, float *mmt,
                 size_t mmtDegree) {
  for (size_t p = 0; p < mmtDegree; p++) {
    for (size_t q = 0; q < mmtDegree; q++) {
      mmt[p + p * q] = 0;

      for (size_t c = 0; c < nc; c++) {
        for (size_t y = 0; y < h; y++) {
          for (size_t x = 0; x < w; x++) {
            /** note: (q+p)th order in the dissertation but not here, 
             *  need to check later 
             */
            mmt[p + (p * q)] +=
                pow(x, p+1) * pow(y, q+1) * imgIn[x + (w * y) + (w * h * c)];
          }
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