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

int getNumForeground(float *imgIn, int w, int h) {
  /** Compute the sum of the coordinates and the number of foreground pixels
     */

  int numForeground = 0;

  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      if (imgIn[x + (w * y)] == FOREGROUND) {
        numForeground++;
      }
    }
  }
  return numForeground;
}

void getCoordForeground(float *imgIn, PixelCoords *pImgIn, int w, int h,
                        PixelCoords *pForeground) {
  // could use vectors to this to append data?
  int index = 0;
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      if (imgIn[x + (w * y)] == FOREGROUND) {
        pForeground[index].x = pImgIn[x + y * w].x;
        pForeground[index].y = pImgIn[x + y * w].y;
        index++;
      }
    }
  }
}

void centerOfMass(float *imgIn, int w, int h, float &xCentCoord,
                  float &yCentCoord) {
  /** Initializations
  * xCentCoord, yCentCoord: x,y indices of the center of mass
  * numOfForegroundPixel: number of the  pixels */
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
                          float yCentCoord, float &normXFactor, float &normYFactor) {
  // Scaling factors per x,y (1/sx, 1/sy in the Matlab implementation)
  normXFactor = 0.5 / max(xCentCoord, w - xCentCoord);
  normYFactor = 0.5 / max(yCentCoord, h - yCentCoord);

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
                          float yCentCoord, float &normXFactor, float &normYFactor) {
  // Scaling factors per x,y (1/sx, 1/sy in the Matlab implementation)
  normXFactor = 0.5 / max(xCentCoord, w - xCentCoord);
  normYFactor = 0.5 / max(yCentCoord, h - yCentCoord);

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

void imageMoment(PixelCoords *pImg, int lenForeground, float *mmt,
                 int mmtDegree) {
  // Compute all the combinations of the (p+q)-order image moments
  // Keep in mind that p,q go from 0 to mmtDegree-1.
  for (int q = 0; q < mmtDegree; q++) {
    for (int p = 0; p < mmtDegree; p++) {
      int mmtIndex = p + q * mmtDegree;
      // cout << "first for set: " << p << " " << q << endl;

      // Compute the image moments taking the contributions from all the pixels
      for (int index = 0; index < lenForeground; index++) {
        /** note: (q+p)th order in the dissertation but not here,
         *  need to check later
         */

        mmt[mmtIndex  + index] =
            pow(pImg[index].x, p + 1) * pow(pImg[index].y, q + 1);
      }
    }
  }
}

void lmminObjectiveWrapper(const double *par, const int m_dat, const void *data, double *residual, int *userbreak) {

	// The affineParam and the localCoeff are our free variables ("parameters") and
	// need to be packed in an array in order to use the lmmin(). We pass
	// them as *par, but our functions are implemented to use the TPSParams
	// structure. We do the unpacking here.
	TPSParams tpsParams;

	for (int i = 0; i < 6; i++) {
		tpsParams.affineParam[i] = par[i];
	}

  /*printf("affineParam[0] = %f, [1] = %f, [2] = %f\n", tpsParams.affineParam[0], tpsParams.affineParam[1], tpsParams.affineParam[2]);*/
  /*printf("affineParam[3] = %f, [4] = %f, [5] = %f\n", tpsParams.affineParam[3], tpsParams.affineParam[4], tpsParams.affineParam[5]);*/

	for (int i = 0; i < 2 * DIM_C_REF * DIM_C_REF; i++) {
		tpsParams.localCoeff[i] = par[i+6];
    // printf("localCoeff[i] = %f\n", tpsParams.localCoeff[i]);
	}

  // printf("tpsParams affine first: %f, last: %f\n", tpsParams.affineParam[0], tpsParams.affineParam[5]);
  // printf("tpsParams localC first: %f, last: %f\n", tpsParams.localCoeff[0], tpsParams.localCoeff[2 * DIM_C_REF * DIM_C_REF - 1]);

  // Cast the void pointer data to a float pointer dataF
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

  // printf("rt_w = %d, rt_h = %d, ro_w = %d, ro_h = %d\n", rt_w, rt_h, ro_w, ro_h);

  // Template image array
	float *templateImg = new float[rt_w * rt_h];
	for (int i = 0; i < rt_w * rt_h; i++) {
		templateImg[i] = dataF[offset + i];
	}
	offset += rt_w * rt_h;

  // printf("templateImg first = %f, last = %f\n", templateImg[0], templateImg[rt_w * rt_h - 1]);

  // Observation image array
	float *observationImg = new float[ro_w * ro_h];
	for (int i = 0; i < ro_w * ro_h; i++) {
		observationImg[i] = dataF[offset + i];
	}
	offset += ro_w * ro_h;

  // printf("observationImg first = %f, last = %f\n", observationImg[0], observationImg[rt_w * rt_h - 1]);

  // Normalization factors (N_i for eq.22)
  double normalization[81]; // TODO: Make this double everywhere
  for (int i = 0; i < 81; i++) {
    normalization[i] = dataF[offset + i];
  }
  offset += 81;

  // printf("normalization first = %f, last = %f\n", normalization[0], normalization[80]);

  // Pixel coordinates of the template
  // Every element is a struct with two fields: x, y
  PixelCoords *pTemplate = new PixelCoords[rt_w * rt_h];
  for (int i = 0; i < rt_w * rt_h; i++) {
    pTemplate[i].x = dataF[offset + 2*i];
    pTemplate[i].y = dataF[offset + 2*i+1];
  }
  offset += 2 * rt_w * rt_h;

  // printf("pTemplate first.x = %f, first.y = %f, last.x = %f, last.y = %f\n", pTemplate[0].x, pTemplate[0].y, pTemplate[rt_w * rt_h-1].x, pTemplate[rt_w * rt_h-1].y);

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

  // printf("qTemplate first.x[0] = %f, first.y[3] = %f, last.x[0] = %f, last.y[3] = %f\n", qTemplate[0].x[0], qTemplate[0].y[3], qTemplate[rt_w * rt_h-1].x[0], qTemplate[rt_w * rt_h-1].y[3]);

  // Pixel coordinates of the observation
  // Every element is a struct with two fields: x, y
  PixelCoords *pObservation = new PixelCoords[ro_w * ro_h];
  for (int i = 0; i < ro_w * ro_h; i++) {
    pObservation[i].x = dataF[offset + 2*i];
    pObservation[i].y = dataF[offset + 2*i+1];
  }
  offset += 2 * ro_w * ro_h;

  // Normalisation factors of the template
  float t_sx, t_sy;
  t_sx = dataF[offset    ];
  t_sy = dataF[offset + 1];
  offset += 2;

  // Normalisation factors of the observation
  float o_sx, o_sy;
  o_sx = dataF[offset    ];
  o_sy = dataF[offset + 1];
  offset += 2;

  // printf("pObservation first.x = %f, last.y = %f\n", pObservation[0].x, pObservation[ro_w * ro_h -1].y);

  // Array of the residuals of the equations
  // TODO: Add also the 6 extra equations!
  // printf("residual first = %f, last = %f\n", residual[0], residual[80]);

  // Call the objective function with the unpacked arguments
  objectiveFunction(observationImg, templateImg, ro_w, ro_h,
                    normalization, tpsParams, qTemplate, pTemplate,
                    pObservation, rt_w, rt_h, t_sx, t_sy, o_sx, o_sy, residual);

  // printf("residual first = %f, last = %f\n", residual[0], residual[80]);

  // Delete the allocated pointers
  delete templateImg;
  delete observationImg;
  delete pTemplate;
  delete qTemplate;

  return;
}


void objectiveFunction(float *observationImg, float *templateImg,
                        int ro_w, int ro_h,
                        double *normalisation, TPSParams &tpsParams,
                        QuadCoords *qTemplate, PixelCoords *pTemplate,
                        PixelCoords *pObservation, int rt_w, int rt_h,
                        float t_sx, float t_sy, float o_sx, float o_sy,
                        double *residual) {
  int momentDeg = 9;
  float resNorm = 0;

  float sumTempMoment[momentDeg * momentDeg] ;
  float sumObsMoment[momentDeg * momentDeg] ;
  //init vars
  for ( int init = 0; init < momentDeg * momentDeg; init ++){
	  sumObsMoment[init] =(float)0;
	  sumTempMoment[init] = (float)0;
  }

  //reszied pTemplate and pObservation to just foreground
  //TODO do for both
  int o_lenForeground;
  int t_lenForeground;

  o_lenForeground = getNumForeground(observationImg , ro_w, ro_h) ;
  t_lenForeground = getNumForeground(templateImg, rt_w, rt_h);

  PixelCoords * pfTemplate = new PixelCoords[t_lenForeground];
  PixelCoords * pfObservation = new PixelCoords[o_lenForeground];

  float * observationMoment = new float[momentDeg * momentDeg * o_lenForeground];
  float * templateMoment= new float[momentDeg * momentDeg * t_lenForeground];

  getCoordForeground(observationImg, pObservation,ro_w, ro_h,
                        pfObservation) ;
  getCoordForeground(templateImg, pTemplate, rt_w,  rt_h,
                        pfTemplate) ;
  

  // get the jacobian at each pixel with the current tps params
  float jacobi[t_lenForeground];
  jacobianTrans(t_lenForeground, jacobi, pfTemplate, tpsParams, DIM_C_REF);

  // calculate tps transformation of template
  pTPS(t_lenForeground, pfTemplate, tpsParams, DIM_C_REF);

  // get the moments of the TPS transformation of the template
  //TODO this is whhere segfaut is
  imageMoment(pfTemplate, t_lenForeground, templateMoment, momentDeg);
  // get the moments of the observation
  imageMoment(pfObservation, o_lenForeground, observationMoment, momentDeg);

  // Determinant of the normFactor of the normalized template image
  float detN1 = 0;
  detN1 = t_sx * t_sy;
  // Determinant of the normFactor of the normalized observation image
  float detN2 = 0;
  detN2 = o_sx * o_sy;

  // Sum the moments of each degree for each pixel of the two images
  for (int index = 0; index < momentDeg * momentDeg; index++) {
    // Transformed template

    // TOOD change params to fit new size
    for (int tempIndex = 0; tempIndex < t_lenForeground; tempIndex++) {
      sumTempMoment[index] +=
          templateMoment[index * t_lenForeground + tempIndex] *
          jacobi[tempIndex];
    }
    sumTempMoment[index] /= detN2;

    // Observation
    for (int obsIndex = 0; obsIndex < o_lenForeground; obsIndex++) {
      sumObsMoment[index] +=
          observationMoment[index * o_lenForeground + obsIndex];
    }

    sumObsMoment[index] /= detN1;
	

    // Compute the residual as the difference between the LHS and the RHS of
    // eq.22
    residual[index] =
        (sumObsMoment[index] - sumTempMoment[index]) / normalisation[index];

    // Residual norm^2 (only for output purposes)
    resNorm += residual[index] * residual[index];
  }

  // First restriction of eq.16 (2 equations)
  int index = momentDeg * momentDeg;
  residual[index] = 0;
  residual[index+1] = 0;
  int K = DIM_C_REF*DIM_C_REF;
  for (int k = 0; k < K; k++) {
    residual[index]   += tpsParams.localCoeff[k];
    residual[index+1] += tpsParams.localCoeff[k + K];
  }
  resNorm += residual[index] * residual[index];
  resNorm += residual[index+1] * residual[index+1];

  index += 2;
  // Second restriction of eq.16 (4 equations)
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      residual[index + (i + 2*j)] = 0;
      for (int k = 0; k < K; k++) {
        residual[index + (i + 2*j)] += tpsParams.ctrlP[k + j*K] * tpsParams.localCoeff[k + i*K];
      }
      resNorm += residual[index + (i + 2*j)] * residual[index + (i + 2*j)];
    }
  }

  // Print the residual norm
  resNorm = sqrt(resNorm);
  printf("Residual norm = %f\n", resNorm);

  /*printf("affineParam[0] = %f, [1] = %f, [2] = %f\n", tpsParams.affineParam[0], tpsParams.affineParam[1], tpsParams.affineParam[2]);*/
  /*printf("affineParam[3] = %f, [4] = %f, [5] = %f\n", tpsParams.affineParam[3], tpsParams.affineParam[4], tpsParams.affineParam[5]);*/

  delete[] observationMoment;
  delete[] templateMoment;
};

// PixelCoords* pCoordsSigma
void pTPS(int lenForeground, PixelCoords *pCoords, TPSParams &tpsParams, int c_dim) {

  int dimSize = c_dim * c_dim;
  float Q;

  //TODO make free defomation subfunction?
 float freeDeformation[2] = {0, 0};
  // for every pixel location
    for (int index = 0; index < lenForeground; index++) {

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

void jacobianTrans(int lenForeground, float *jacobi, PixelCoords * pCoords,
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
  for (indexP = 0; indexP < lenForeground; indexP++) {
    // Reset the local jacobi elements to a_ij for the current pixel
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        indexJ = i + 2 * j;
        jacEl[indexJ] = tpsParams.affineParam[i + 3 * j];
      }
    }

    // Note: synchronize here in the GPU version

    // For each control point
    for (int k = 0; k < K; k++) {
      // Compute the argument of the log()
      // squareOfNorm = (ck_x - x)^2 + (ck_y - y)^2
      squareOfNorm = (tpsParams.ctrlP[k] - pCoords[indexP].x) *
                         (tpsParams.ctrlP[k] - pCoords[indexP].x) +
                     (tpsParams.ctrlP[k + K] - pCoords[indexP].y) *
                         (tpsParams.ctrlP[k + K] - pCoords[indexP].y);
      // TODO this should be globaly defined as eps
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
          indexJ = i + 2 * j;
          // Do we need the x or the y in the place of x_j?
          x_j = (j == 0 ? pCoords[indexP].x : pCoords[indexP].y);
          // jacobi_ij -= precomp * w_ki * (c_kj - x_j)
          jacEl[indexJ] -= precomp * tpsParams.localCoeff[k + i * K] *
                           (tpsParams.ctrlP[k + j * K] - x_j);
        }
      }
    // Compute the determinant of the local jacobi elements
    jacobi[indexP] = jacEl[0] * jacEl[3] - jacEl[1] * jacEl[2];
    }

  }
return;
}

  void transfer(float *imgOut, PixelCoords *pCoords, int o_w, int o_h,
                float *imgIn, QuadCoords *qCoords, int t_w, int t_h) {
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
