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
void pCoordsNormalisation(int w, int h, PixelCoords *pCoords, float xCentCoord,
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


void imageMoment(float *imgIn, PixelCoords *pImg, int w, int h, float *mmt,
                 int mmtDegree) {
  // Compute all the combinations of the (p+q)-order image moments
  // Keep in mind that p,q go from 0 to mmtDegree-1.
  for (int q = 0; q < mmtDegree; q++) {
    for (int p = 0; p < mmtDegree; p++) {
      int mmtIndex = p + q * mmtDegree;
	  //cout << "first for set: " << p << " " << q << endl;

      // Compute the image moments taking the contributions from all the pixels
      for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
          int index = x + (w * y);
          /** note: (q+p)th order in the dissertation but not here,
           *  need to check later
           */

          mmt[mmtIndex * ( w * h) + index] = pow(pImg[index].x, p + 1) *
                                      pow(pImg[index].y, q + 1) * imgIn[index];
        }
      }
    }
  }
}

void lmminObjectiveWrapper(const double *par, const int m_dat, const void *data, double *fvec, int *userbreak) {

	// The affineParam and the localCoeff are our free variables ("parameters") and
	// need to be packed in an array in order to use the lmmin(). We pass
	// them as *par, but our functions are implemented to use the TPSParams
	// structure. We do the unpacking here.
	TPSParams tpsParams;

	for (int i = 0; i < 6; i++) {
		tpsParams.affineParam[i] = par[i];
	}

	for (int i = 0; i < 2 * DIM_C_REF * DIM_C_REF; i++) {
		tpsParams.localCoeff[i] = par[i+6];
	}

  // Cast the void pointer data to a float pointer dataF
  // void *dataF = data;
  // void *dataF = data;
  // float *dataF = reinterpret_cast<float*>(data);
  const float *dataF = static_cast<const float *>(data);

	// We also need to pack/unpack the non-free parameters ("data") of the objective function
  // current reading position in the data array
	int offset = 0;

  // Read first the sizes needed to allocate the included arrays
  // int rt_w = static_cast<int>(data[offset]);
	int rt_w = dataF[offset    ];
	int rt_h = dataF[offset + 1];
	int ro_w = dataF[offset + 2];
	int ro_h = dataF[offset + 3];
  // We read 4 elements, move the reading position 4 places
	offset += 4;

  // Template image array
	float *templateImg = new float[rt_w * rt_h];
	for (int i = 0; i < rt_w * rt_h; i++) {
		templateImg[i] = dataF[offset + i];
	}
	offset += rt_w * rt_h;

  // Observation image array
	float *observationImg = new float[ro_w * ro_h];
	for (int i = 0; i < ro_w * ro_h; i++) {
		observationImg[i] = dataF[offset + i];
	}
	offset += ro_w * ro_h;

  // Normalization factors (N_i for eq.22)
  double normalization[81]; // TODO: Make this double everywhere
  for (int i = 0; i < 81; i++) {
    normalization[i] = dataF[offset + i];
  }
  offset += 81;

  // Pixel coordinates of the template
  // Every element is a struct with two fields: x, y
  PixelCoords *pTemplate = new PixelCoords[rt_w * rt_h];
  for (int i = 0; i < rt_w * rt_h; i++) {
    pTemplate[i].x = dataF[offset + 2*i];
    pTemplate[i].y = dataF[offset + 2*i+1];
  }
  offset += 2 * rt_w * rt_h;

  // Quad coordinates of the template
  // Every element has two fields (x,y) that are arrays of four elements (corners)
  QuadCoords *qTemplate = new QuadCoords[rt_w * rt_h];
  for (int i = 0; i < rt_w * rt_h; i++) {
    qTemplate[i].x[0] = dataF[offset + 8*i  ];
    qTemplate[i].y[0] = dataF[offset + 8*i+1];
    qTemplate[i].x[1] = dataF[offset + 8*i+2];
    qTemplate[i].y[1] = dataF[offset + 8*i+3];
    qTemplate[i].x[2] = dataF[offset + 8*i+4];
    qTemplate[i].y[2] = dataF[offset + 8*i+5];
    qTemplate[i].x[3] = dataF[offset + 8*i+6];
    qTemplate[i].y[3] = dataF[offset + 8*i+7];
  }
  offset += 8 * rt_w * rt_h;

  // Pixel coordinates of the observation
  // Every element is a struct with two fields: x, y
  PixelCoords *pObservation = new PixelCoords[ro_w * ro_h];
  for (int i = 0; i < ro_w * ro_h; i++) {
    pObservation[i].x = dataF[offset + 2*i];
    pObservation[i].y = dataF[offset + 2*i+1];
  }
  offset += 2 * ro_w * ro_h;

  // Jacobi determinants
  float *jacobi = new float[rt_w * rt_h];
  for (int i = 0; i < rt_w * rt_h; i++) {
    jacobi[i] = dataF[offset + i];
  }
  offset += rt_w * rt_h;

  // Array of the residuals of the equations
  // TODO: Add also the 6 extra equations!
  float residual[9 * 9] = { };
  for (int i = 0; i < 9 * 9; i++) {
    residual[i] = fvec[i];
  }

  // Call the objective function with the unpacked arguments
  objectiveFunction(observationImg, templateImg, jacobi, ro_w, ro_h,
                    normalization, tpsParams, qTemplate, pTemplate,
                    pObservation, rt_w, rt_h, residual);

  // Delete the allocated pointers
  delete templateImg;
  delete observationImg;
  delete pTemplate;
  delete qTemplate;
  delete jacobi;

  return;
}


void objectiveFunction(float *observationImg, float *templateImg,
                        float *jacobi, int ro_w, int ro_h,
                        double *normalisation, TPSParams &tpsParams,
                        QuadCoords *qTemplate, PixelCoords *pTemplate,
                        PixelCoords *pObservation, int rt_w, int rt_h,
                        float *residual) {
  int momentDeg = 9;

  float * observationMoment = new float[momentDeg * momentDeg * ro_w * ro_h];
  float * templateMoment= new float[momentDeg * momentDeg * rt_w * rt_h];


  float sumTempMoment[momentDeg * momentDeg] ;
  float sumObsMoment[momentDeg * momentDeg] ;
  for ( int init = 0; init < momentDeg * momentDeg; init ++){
	  sumObsMoment[init] =(float)0;
	  sumTempMoment[init] = (float)0;
  }


  // calculate tps transformation of template

  qTPS(rt_w, rt_h, qTemplate, tpsParams, DIM_C_REF);

  transfer(templateImg, pObservation, qTemplate, rt_w, rt_h, ro_w, ro_h,
           observationImg);

  // get moments of TPS transformation of template
  imageMoment(observationImg, pObservation, ro_w, ro_h, observationMoment,
			  momentDeg);

  imageMoment(templateImg, pTemplate, rt_w, rt_h, templateMoment, momentDeg);

  // get jacobian of current tps params
  jacobianTrans(rt_w, rt_h, jacobi, pTemplate, tpsParams, DIM_C_REF);
  // get determinant of Jacobian

  for (int index = 0; index < momentDeg * momentDeg; index++) {
    for (int y = 0; y < rt_h; y++) {
      for (int x = 0; x < rt_w; x++) {
        sumTempMoment[index] +=
            templateMoment[index * (rt_h * rt_w) + (x + rt_w * y)] *
            jacobi[x + rt_w * y];
      }
    }

    for (int y = 0; y < ro_h; y++) {
      for (int x = 0; x < ro_w; x++) {
        sumObsMoment[index] +=
            observationMoment[index * (ro_h * ro_w) + (x + ro_w * y)];
      }
    }

    residual[index] =
        (sumObsMoment[index] - sumTempMoment[index]) / normalisation[index];


  }
  delete[] observationMoment;
  delete[] templateMoment;
};

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

          /**   (a_i1 *x_1)  + (a_i2 *x_2) + a_i3
           *  = (scale*x_1)  + (sheer*x_2) + translation
           *  = (        rotation        ) + translation
           */

          // multiply with weights
          for (int i = 0; i < 2; i++) {
            freeDeformation[i] += tpsParams.localCoeff[k + (i * dimSize)] * Q;
          }
        }

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

void jacobianTrans(int w, int h, float *jacobi, PixelCoords * pCoords,
                   TPSParams &tpsParams, int c_dim) {
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
        squareOfNorm = (tpsParams.ctrlP[k] - pCoords[indexP].x)   * (tpsParams.ctrlP[k] - pCoords[indexP].x)
                     + (tpsParams.ctrlP[k+K] - pCoords[indexP].y) * (tpsParams.ctrlP[k+K] - pCoords[indexP].y);
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
            x_j = (j == 0 ? pCoords[indexP].x : pCoords[indexP].y);
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
