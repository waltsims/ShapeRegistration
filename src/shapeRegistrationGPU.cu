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

#include "helper.h"
#include "shapeRegistrationGPU.h"
#include <stdio.h>

__global__ void setPixelCoordsKernel(PixelCoords *d_pCoords, int d_w, int d_h) {
  int index;

  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;

  if (x < d_w && y < d_h) {
    index = x + y * d_w;
    d_pCoords[index].x = (float)y;
    d_pCoords[index].y = (float)x;
  }

}

void setPixelCoordsGPU(PixelCoords *h_pCoords, int h_w, int h_h) {
  dim3 block = dim3(128, 1, 1); 
  dim3 grid = dim3((h_w + block.x - 1) / block.x, (h_h + block.y - 1) / block.y,
                   1);

  PixelCoords *d_pCoords;
  cudaMalloc(&d_pCoords, h_w * h_h * sizeof(PixelCoords));
  CUDA_CHECK;

  cudaMemcpy(d_pCoords, h_pCoords, h_w * h_h * sizeof(PixelCoords),
             cudaMemcpyHostToDevice);
  CUDA_CHECK;

  setPixelCoordsKernel << <grid, block>>> (d_pCoords, h_w, h_h);

  cudaMemcpy(h_pCoords, d_pCoords, h_w * h_h * sizeof(PixelCoords),
             cudaMemcpyDeviceToHost);
  CUDA_CHECK;

  cudaFree(d_pCoords);
  CUDA_CHECK;
}

__global__ void imageMomentKernel(float *d_imgIn, PixelCoords *d_pImg, int d_w,
                                  int d_h, float *d_mmt, int d_mmtDegree) {
  int index;

  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;

  if (x < d_w && y < d_h) {
    index = x + y * d_w;
    // Compute all the combinations of the (p+q)-order image moments
    // Keep in mind that p,q go from 0 to mmtDegree-1.

    for (int q = 0; q < d_mmtDegree; q++) {
      for (int p = 0; p < d_mmtDegree; p++) {
        int mmtIndex = p + q * d_mmtDegree;

        // Compute the image moments taking the contributions from all the
        // pixels

        d_mmt[mmtIndex * (d_w * d_h) + index] = pow(d_pImg[index].x, p + 1) *
                                                pow(d_pImg[index].y, q + 1) *
                                                d_imgIn[index];
      }
    }
  }
}

void imageMomentGPU(float *h_imgIn, PixelCoords *h_pImg, int h_w, int h_h,
                    float *h_mmt, int h_mmtDegree) {

  dim3 block = dim3(128, 1, 1); 
  dim3 grid = dim3((h_w + block.x - 1) / block.x, (h_h + block.y - 1) / block.y,
                   1);

  PixelCoords *d_pImg;
  cudaMalloc(&d_pImg, h_w * h_h * sizeof(PixelCoords));
  CUDA_CHECK;

  float *d_imgIn;
  cudaMalloc(&d_imgIn, h_w * h_h * sizeof(float));
  CUDA_CHECK;

  float *d_mmt;
  cudaMalloc(&d_mmt, h_mmtDegree * h_mmtDegree * h_w * h_h * sizeof(float));
  CUDA_CHECK;

  cudaMemcpy(d_imgIn, h_imgIn, h_w * h_h * sizeof(float), cudaMemcpyHostToDevice);CUDA_CHECK;
  cudaMemcpy(d_pImg, h_pImg, h_w * h_h * sizeof(PixelCoords), cudaMemcpyHostToDevice);CUDA_CHECK;

  imageMomentKernel <<<grid, block>>> (d_imgIn, d_pImg, h_w, h_h, d_mmt, h_mmtDegree);

  cudaMemcpy(h_mmt, d_mmt, h_mmtDegree * h_mmtDegree * h_w * h_h * sizeof(float), cudaMemcpyDeviceToHost);
  

  cudaFree(d_imgIn);
  CUDA_CHECK;
  cudaFree(d_mmt);
  CUDA_CHECK;
  cudaFree(d_pImg);
  CUDA_CHECK;
  
}

__global__ void setQuadCoordsKernel(QuadCoords *d_qCoords, int d_w, int d_h) {
  int index;

  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;

  if (x < d_w && y < d_h) {
    index = x + y * d_w;
    d_qCoords[index].x[0] = (float)x - 0.5;
    d_qCoords[index].y[0] = (float)y - 0.5;
    d_qCoords[index].x[1] = (float)x + 0.5;
    d_qCoords[index].y[1] = (float)y - 0.5;
    d_qCoords[index].x[2] = (float)x + 0.5;
    d_qCoords[index].y[2] = (float)y + 0.5;
    d_qCoords[index].x[3] = (float)x - 0.5;
    d_qCoords[index].y[3] = (float)y + 0.5;
  }
}

void setQuadCoordsGPU(QuadCoords *h_qCoords, int h_w, int h_h) {
  dim3 block = dim3(128, 1, 1);
  dim3 grid = dim3((h_w + block.x - 1) / block.x, (h_h + block.y - 1) / block.y,
                   1); 

  QuadCoords *d_qCoords;
  cudaMalloc(&d_qCoords, h_w * h_h * sizeof(QuadCoords));
  CUDA_CHECK;

  cudaMemcpy(d_qCoords, h_qCoords, h_w * h_h * sizeof(QuadCoords),
             cudaMemcpyHostToDevice);
  CUDA_CHECK;

  setQuadCoordsKernel << <grid, block>>> (d_qCoords, h_w, h_h);

  cudaMemcpy(h_qCoords, d_qCoords, h_w * h_h * sizeof(QuadCoords),
             cudaMemcpyDeviceToHost);

  cudaFree(d_qCoords);
  CUDA_CHECK;
}


void cutMarginsGPU(float *imgIn, int w, int h, float *&resizedImg,
                   int &resizedW, int &resizedH, Margins &margins) {
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

__global__ void addMarginsKernel(float *d_resizedImg, int d_resizedW,
                                 int d_resizedH, float *d_imgOut, int d_w,
                                 int d_h, Margins &d_margins) {

  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  if (x < d_w && y < d_h) {
    int index = x + y * d_w;
    int res_index = (x - d_margins.left) + (y - d_margins.top) * d_resizedW;

    if (x >= d_margins.left && x <= d_margins.right && y >= d_margins.top &&
        y <= d_margins.bottom) {
      d_imgOut[index] = d_resizedImg[res_index];
    } else {
      d_imgOut[index] = BACKGROUND;
    }
  }
}

void addMarginsGPU(float *resizedImg, int resizedW, int resizedH, float *imgOut,
                   int w, int h, Margins &margins) {
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      int index = x + y * w;
      int res_index = (x - margins.left) + (y - margins.top) * resizedW;
      if (x >= margins.left && x <= margins.right && y >= margins.top &&
          y <= margins.bottom) {
        imgOut[index] = resizedImg[res_index];
      } else {
        imgOut[index] = BACKGROUND;
      }
    }
  }
}

__global__ void centerOfMassKernel(float *d_imgIn, int d_w, int d_h,
                                   float *d_xCentCoord, float *d_yCentCoord) {
  /*
    int index;

    // Initializations
    // xCentCoord, yCentCoord: x,y indices of the center of mass
    // numOfForegroundPixel: number of the foreground pixels

    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int numOfForegroundPixel = 0;

    if (x < d_w && y < d_h) {
      index = x + (d_w * y);
      if (d_imgIn[index] == FOREGROUND) {
        d_xCentCoord += x;
        d_yCentCoord += y;
        atomicAdd(&numOfForegroundPixel, 1);
      }
    }
    // Average: divide the sum of the coordinates to the number of pixels
    d_xCentCoord /= numOfForegroundPixel;
    d_yCentCoord /= numOfForegroundPixel;
  */

    int index;

    // Initializations
    // xCentCoord, yCentCoord: x,y indices of the center of mass
    // numOfForegroundPixel: number of the foreground pixels

    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    int t_x = threadIdx.x;
    int t_y = threadIdx.y;

    extern __shared__ int numOfForegroundPixel[];

    if (x < d_w && y < d_h) {
      index = x + (d_w * y);
      numOfForegroundPixel[127] = 0;
      if (d_imgIn[index] == FOREGROUND) {
/*        atomicAdd(d_xCentCoord, x);
        atomicAdd(d_yCentCoord, y);*/
        //numOfForegroundPixel[index]++;
      }
    }

    __syncthreads();
/*
    if (x < d_w && y < d_h) {
      index = x + (d_w * y);
      for (int offset = blockDim.x * blockDim.y / 2; offset > 0 ; offset /= 2) {
        if (index < offset){
              numOfForegroundPixel[index] += numOfForegroundPixel[index +
    offset];
        }
        __syncthreads();
      }
    }

    __syncthreads();

    if (x < d_w && y < d_h) {
      index = x + (d_w * y);
      for (int offset = blockDim.x * blockDim.y / 2; offset > 0 ; offset /= 2) {
        if (index < offset){
              numOfForegroundPixel[index] += numOfForegroundPixel[index +
    offset];
        }
        __syncthreads();
      }
    }

    if (t_x == 0 && t_y == 0) {
      // Average: divide the sum of the coordinates to the number of pixels
      d_xCentCoord /= numOfForegroundPixel[0];
      d_yCentCoord /= numOfForegroundPixel[0];
    }*/
}

void centerOfMassGPU(float *h_imgIn, int h_w, int h_h, float &h_xCentCoord,
                     float &h_yCentCoord) {
  h_xCentCoord = 0;
  h_yCentCoord = 0;
  int numOfForegroundPixel = 0;

  /** Compute the sum of the coordinates and the number of foreground pixels */
  for (int y = 0; y < h_h; y++) {
    for (int x = 0; x < h_w; x++) {
      if (h_imgIn[x + (h_w * y)] == FOREGROUND) {
        h_xCentCoord += x;
        h_yCentCoord += y;
        numOfForegroundPixel++;
      }
    }
  }

  /** Average: divide the sum of the coordinates to the number of pixels */
  h_xCentCoord /= numOfForegroundPixel;
  h_yCentCoord /= numOfForegroundPixel;
}

__global__ void pCoordsNormalizationKernel(int d_w, int d_h,
                                           PixelCoords *d_pCoords,
                                           float d_xCentCoord,
                                           float d_yCentCoord) {
  int index;

  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;

  float normXFactor = 0.5 / max(d_xCentCoord, d_w - d_xCentCoord);
  float normYFactor = 0.5 / max(d_yCentCoord, d_h - d_yCentCoord);

  if (x < d_w && y < d_h) {
    index = x + y * d_w;

    d_pCoords[index].x = (d_pCoords[index].x - d_xCentCoord) * normXFactor;
    d_pCoords[index].y = (d_pCoords[index].y - d_yCentCoord) * normYFactor;
  }
}

void pCoordsNormalizationGPU(int h_w, int h_h, PixelCoords *h_pCoords,
                             float h_xCentCoord, float h_yCentCoord) {
  dim3 block = dim3(128, 1, 1);
  dim3 grid = dim3((h_w + block.x - 1) / block.x, (h_h + block.y - 1) / block.y,
                   1);

  PixelCoords *d_pCoords;
  cudaMalloc(&d_pCoords, h_w * h_h * sizeof(PixelCoords));
  CUDA_CHECK;

  cudaMemcpy(d_pCoords, h_pCoords, h_w * h_h * sizeof(PixelCoords),
             cudaMemcpyHostToDevice);
  CUDA_CHECK;

  pCoordsNormalizationKernel << <grid, block>>>
      (h_w, h_h, d_pCoords, h_xCentCoord, h_yCentCoord);

  cudaMemcpy(h_pCoords, d_pCoords, h_w * h_h * sizeof(PixelCoords),
             cudaMemcpyDeviceToHost);
  CUDA_CHECK;

  cudaFree(d_pCoords);
  CUDA_CHECK;
}

__global__ void qCoordsNormalizationKernel(int d_w, int d_h,
                                           QuadCoords *d_qCoords,
                                           float d_xCentCoord,
                                           float d_yCentCoord) {
  int index;

  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;

  float normXFactor = 0.5 / max(d_xCentCoord, d_w - d_xCentCoord);
  float normYFactor = 0.5 / max(d_yCentCoord, d_h - d_yCentCoord);

  if (x < d_w && y < d_h) {
    index = x + y * d_w;
    d_qCoords[index].x[0] =
        (d_qCoords[index].x[0] - d_xCentCoord) * normXFactor;
    d_qCoords[index].y[0] =
        (d_qCoords[index].y[0] - d_yCentCoord) * normYFactor;
    d_qCoords[index].x[1] =
        (d_qCoords[index].x[1] - d_xCentCoord) * normXFactor;
    d_qCoords[index].y[1] =
        (d_qCoords[index].y[1] - d_yCentCoord) * normYFactor;
    d_qCoords[index].x[2] =
        (d_qCoords[index].x[2] - d_xCentCoord) * normXFactor;
    d_qCoords[index].y[2] =
        (d_qCoords[index].y[2] - d_yCentCoord) * normYFactor;
    d_qCoords[index].x[3] =
        (d_qCoords[index].x[3] - d_xCentCoord) * normXFactor;
    d_qCoords[index].y[3] =
        (d_qCoords[index].y[3] - d_yCentCoord) * normYFactor;
  }
}

void qCoordsNormalizationGPU(int h_w, int h_h, QuadCoords *h_qCoords,
                             float h_xCentCoord, float h_yCentCoord) {
  dim3 block = dim3(128, 1, 1);
  dim3 grid = dim3((h_w + block.x - 1) / block.x, (h_h + block.y - 1) / block.y,
                   1);

  QuadCoords *d_qCoords;
  cudaMalloc(&d_qCoords, h_w * h_h * sizeof(QuadCoords));
  CUDA_CHECK;

  cudaMemcpy(d_qCoords, h_qCoords, h_w * h_h * sizeof(QuadCoords),
             cudaMemcpyHostToDevice);
  CUDA_CHECK;

  qCoordsNormalizationKernel << <grid, block>>>
      (h_w, h_h, d_qCoords, h_xCentCoord, h_yCentCoord);

  cudaMemcpy(h_qCoords, d_qCoords, h_w * h_h * sizeof(QuadCoords),
             cudaMemcpyDeviceToHost);
  CUDA_CHECK;

  cudaFree(d_qCoords);
  CUDA_CHECK;
}

__device__ float radialApproxKernel(float d_x, float d_y, float d_cx, float d_cy) {
  float d_r2 = (d_cx - d_x) * (d_cx - d_x) + (d_cy - d_y) * (d_cy - d_y);

  return d_r2 < 0.0000000001 ? 0 : d_r2 * log(d_r2);
}


// PixelCoords* pCoordsSigma
__global__ void pTPSGPUKernel(int d_w, int d_h, PixelCoords *d_pCoords,
                              TPSParams d_tpsParams, int d_cDim) {
  int index;

  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;

  index = x + d_w * y;
  int dimSize = d_cDim * d_cDim;
  float Q;
  float freeDeformation[2] = {0, 0};

  if (x < d_w && y < d_h) {
    Q = 0;
    freeDeformation[0] = 0;
    freeDeformation[1] = 0;
    // for all c_m support coordinates
    for (int k = 0; k < dimSize; k++) {
      // calculate radial approximation

      Q = radialApproxKernel(d_pCoords[index].x,
                             d_pCoords[index].y, d_tpsParams.ctrlP[k],
                             d_tpsParams.ctrlP[k + dimSize]);

      // multiply with weights
      for (int i = 0; i < 2; i++) {
        freeDeformation[i] += d_tpsParams.localCoeff[k + (i * dimSize)] * Q;
      }
    }

    // note:: change
    float tempQCoordsX = d_pCoords[index].x;
    float tempQCoordsY = d_pCoords[index].y;

    d_pCoords[index].x = (d_tpsParams.affineParam[0] * tempQCoordsX) +
                                 (d_tpsParams.affineParam[1] * tempQCoordsY) +
                                 d_tpsParams.affineParam[2] +
                                 freeDeformation[0];

    d_pCoords[index].y = (d_tpsParams.affineParam[3] * tempQCoordsX) +
                                 (d_tpsParams.affineParam[4] * tempQCoordsY) +
                                 d_tpsParams.affineParam[5] +
                                 freeDeformation[1];
  }
}

void pTPSGPU(int h_w, int h_h, PixelCoords *h_pCoords, TPSParams &h_tpsParams,
             int h_cDim) {

  dim3 block = dim3(128, 1, 1);
  dim3 grid =
      dim3((h_w + block.x - 1) / block.x, (h_h + block.y - 1) / block.y, 1);


  PixelCoords *d_pCoords;
  cudaMalloc(&d_pCoords, h_h * h_w * sizeof(PixelCoords));
  CUDA_CHECK;

  cudaMemcpy(d_pCoords, h_pCoords, h_h * h_w * sizeof(PixelCoords),
             cudaMemcpyHostToDevice);
  CUDA_CHECK;

  pTPSGPUKernel<<<grid, block>>>(h_w, h_h, d_pCoords, h_tpsParams, h_cDim);
  CUDA_CHECK;

  cudaMemcpy(h_pCoords, d_pCoords, h_h * h_w * sizeof(PixelCoords),
             cudaMemcpyDeviceToHost);
  CUDA_CHECK;

  cudaFree(d_pCoords);
  CUDA_CHECK;
}


__global__ void qTPSGPUKernel(int d_w, int d_h, QuadCoords *d_qCoords,
                              TPSParams d_tpsParams, int d_cDim) {
  int index;

  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;

  index = x + d_w * y;
  int dimSize = d_cDim * d_cDim;
  float Q;
  float freeDeformation[2] = {0, 0};

  if (x < d_w && y < d_h) {
    for (int qIndex = 0; qIndex < 4; qIndex++) {
      Q = 0;
      freeDeformation[0] = 0;
      freeDeformation[1] = 0;
      // for all c_m support coordinates
      for (int k = 0; k < dimSize; k++) {
        // calculate radial approximation

        Q = radialApproxKernel(d_qCoords[index].x[qIndex],
                               d_qCoords[index].y[qIndex], d_tpsParams.ctrlP[k],
                               d_tpsParams.ctrlP[k + dimSize]);

        // multiply with weights
        for (int i = 0; i < 2; i++) {
          freeDeformation[i] += d_tpsParams.localCoeff[k + (i * dimSize)] * Q;
        }
      }

      // note:: change
      float tempQCoordsX = d_qCoords[index].x[qIndex];
      float tempQCoordsY = d_qCoords[index].y[qIndex];

      d_qCoords[index].x[qIndex] = (d_tpsParams.affineParam[0] * tempQCoordsX) +
                                   (d_tpsParams.affineParam[1] * tempQCoordsY) +
                                   d_tpsParams.affineParam[2] +
                                   freeDeformation[0];

      d_qCoords[index].y[qIndex] = (d_tpsParams.affineParam[3] * tempQCoordsX) +
                                 (d_tpsParams.affineParam[4] * tempQCoordsY) +
                                 d_tpsParams.affineParam[5] + freeDeformation[1];
    }
  }
}

void qTPSGPU(int h_w, int h_h, QuadCoords *h_qCoords, TPSParams &h_tpsParams,
             int h_cDim) {

  dim3 block = dim3(128, 1, 1);
  dim3 grid =
      dim3((h_w + block.x - 1) / block.x, (h_h + block.y - 1) / block.y, 1);


  QuadCoords *d_qCoords;
  cudaMalloc(&d_qCoords, h_h * h_w * sizeof(QuadCoords));
  CUDA_CHECK;

  cudaMemcpy(d_qCoords, h_qCoords, h_h * h_w * sizeof(QuadCoords),
             cudaMemcpyHostToDevice);
  CUDA_CHECK;

  qTPSGPUKernel<<<grid, block>>>(h_w, h_h, d_qCoords, h_tpsParams, h_cDim);
  CUDA_CHECK;

  cudaMemcpy(h_qCoords, d_qCoords, h_h * h_w * sizeof(QuadCoords),
             cudaMemcpyDeviceToHost);
  CUDA_CHECK;

  cudaFree(d_qCoords);
  CUDA_CHECK;
}


float radialApproxGPU(float x, float y, float cx, float cy) {
	//TODO delete if only device function needed
  float r2 = (cx - x) * (cx - x) + (cy - y) * (cy - y);

  return r2 < 0.0000000001 ? 0 : r2 * log(r2);
}

void jacobianTransGPU(int w, int h, float *jacobi, TPSParams &tpsParams,
                      int c_dim) {
  // Index in the *jacobi
  int indexP, indexJ;
  // Number of control points
  int K = c_dim * c_dim;
  // Square of the distance of the control point from the pixel
  float squareOfNorm;
  // Common term for all the i,j in each c_k,x combination (precomputed)
  float precomp;
  // x_j (x or y)
  float x_j;

  // For each pixel
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      // Index of the pixel in the image
      indexP = x + w * y;

      // Reset the jacobi elements to a_ij for the current pixel
      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
          indexJ = 4 * indexP + i + 2 * j;
          jacobi[indexJ] = tpsParams.affineParam[i + 2 * j];
        }
      }

      // Note: synchronize here in the GPU version

      // For each control point
      for (int k = 0; k < K; k++) {
        // Compute the argument of the log()
        // squareOfNorm = (ck_x - x)^2 + (ck_y - y)^2
        squareOfNorm =
            (tpsParams.ctrlP[k] - x) * (tpsParams.ctrlP[k] - x) +
            (tpsParams.ctrlP[k + K] - y) * (tpsParams.ctrlP[k + K] - y);
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
            // Index in the global jacobi array
            indexJ = 4 * indexP + i + 2 * j;
            // Do we need the x or the y in the place of x_j?
            x_j = (j == 0 ? x : y);
            // jacobi_ij -= precomp * w_ki * (c_kj - x_j)
            jacobi[indexJ] -= precomp * tpsParams.localCoeff[k + i * K] *
                              (tpsParams.ctrlP[k + j * K] - x_j);
          }
        }
      }
    }
  }
  return;
}

/*  int index;
  int p_index;

  for (int j = 0; j < t_h; j++) {
    for (int i = 0; i < t_w; i++) {
      index = i + t_w * j;
      if (imgIn[index] == FOREGROUND) {
        float xpolygon[4] = {qCoords[index].x[0], qCoords[index].x[1],
                             qCoords[index].x[2], qCoords[index].x[3]};
        float ypolygon[4] = {qCoords[index].y[0], qCoords[index].y[1],
                             qCoords[index].y[2], qCoords[index].y[3]};



        int xLeftOffset = int(floor(min(qCoords[index].x[0],
  qCoords[index].x[3])));
        int xRightOffset = int(ceil(max(qCoords[index].x[1],
  qCoords[index].x[2])));
        int yTopOffset = int(floor(min(qCoords[index].y[0],
  qCoords[index].y[1])));
        int yBottomOffset = int(ceil(max(qCoords[index].y[2],
  qCoords[index].y[3])));


        printf("------------i, j = %d, %d-------------\n", i, j);
        printf("left, right = %d, %d||", xLeftOffset, xRightOffset);
        printf("top, bottom = %d, %d\n", yTopOffset, yBottomOffset);
        // TODO create local index to search for neignboring points
        // withing bounding box of polygon
        for (int y = yTopOffset; y < yBottomOffset; y++) {
          for (int x = xLeftOffset; x < xRightOffset; x++) {
            p_index = x + o_w * y;

            if (pointInPolygonGPU(4, xpolygon, ypolygon, pCoords[p_index].x,
                               pCoords[p_index].y))
              imgOut[p_index] = FOREGROUND;
          }
        }
      }
    }
  }*/

__device__ bool pointInPolygonKernel(int nVert, float *vertX, float *vertY,
                                     float testX, float testY) {
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

__global__ void transferKernel(float *d_imgIn, PixelCoords *d_pCoords,
                               QuadCoords *d_qCoords, int d_t_w, int d_t_h,
                               int d_o_w, int d_o_h, float *d_imgOut) {
  int index;

  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;

  int p_index;

  if (i < d_t_w && j < d_t_h) {
    index = i + d_t_w * j;
    if (d_imgIn[index] == FOREGROUND) {
      float xpolygon[4] = {d_qCoords[index].x[0], d_qCoords[index].x[1],
                           d_qCoords[index].x[2], d_qCoords[index].x[3]};
      float ypolygon[4] = {d_qCoords[index].y[0], d_qCoords[index].y[1],
                           d_qCoords[index].y[2], d_qCoords[index].y[3]};

      for (int x = 0; x < d_o_w; x++) {
        for (int y = 0; y < d_o_h; y++) {
          p_index = x + d_o_w * y;

          if (pointInPolygonKernel(4, xpolygon, ypolygon, d_pCoords[p_index].x,
                                   d_pCoords[p_index].y))
            d_imgOut[p_index] = FOREGROUND;
        }
      }
    }
  }
}

void transferGPU(float *h_imgIn, PixelCoords *h_pCoords, QuadCoords *h_qCoords,
                 int h_t_w, int h_t_h, int h_o_w, int h_o_h, float *h_imgOut) {
  dim3 block = dim3(128, 1, 1);
  dim3 grid =
      dim3((h_t_w + block.x - 1) / block.x, (h_t_h + block.y - 1) / block.y,
           1);

  float *d_imgIn;
  PixelCoords *d_pCoords;
  QuadCoords *d_qCoords;
  float *d_imgOut;

  cudaMalloc(&d_imgIn, h_t_w * h_t_h * sizeof(float));
  CUDA_CHECK;
  cudaMalloc(&d_pCoords, h_o_w * h_o_h * sizeof(PixelCoords));
  CUDA_CHECK;
  cudaMalloc(&d_qCoords, h_t_w * h_t_h * sizeof(QuadCoords));
  CUDA_CHECK;
  cudaMalloc(&d_imgOut, h_o_w * h_o_h * sizeof(float));
  CUDA_CHECK;

  cudaMemcpy(d_imgIn, h_imgIn, h_t_w * h_t_h * sizeof(float),
             cudaMemcpyHostToDevice);
  CUDA_CHECK;
  cudaMemcpy(d_pCoords, h_pCoords, h_o_w * h_o_h * sizeof(PixelCoords),
             cudaMemcpyHostToDevice);
  CUDA_CHECK;
  cudaMemcpy(d_qCoords, h_qCoords, h_t_w * h_t_h * sizeof(QuadCoords),
             cudaMemcpyHostToDevice);
  CUDA_CHECK;

  cudaMemset(d_imgOut, 0, h_o_w * h_o_h * sizeof(float));
  CUDA_CHECK;

  transferKernel << <grid, block>>>
      (d_imgIn, d_pCoords, d_qCoords, h_t_w, h_t_h, h_o_w, h_o_h, d_imgOut);

/*  cudaMemcpy(h_imgIn, d_imgIn, h_t_w * h_t_h * sizeof(float),
             cudaMemcpyDeviceToHost);
  CUDA_CHECK;
  cudaMemcpy(h_pCoords, d_pCoords, h_o_w * h_o_h * sizeof(PixelCoords),
             cudaMemcpyDeviceToHost);
  CUDA_CHECK;
  cudaMemcpy(h_qCoords, d_qCoords, h_t_w * h_t_h * sizeof(QuadCoords),
             cudaMemcpyDeviceToHost);
  CUDA_CHECK;*/
  cudaMemcpy(h_imgOut, d_imgOut, h_o_w * h_o_h * sizeof(float),
             cudaMemcpyDeviceToHost);
  CUDA_CHECK;

  cudaFree(d_imgIn);
  CUDA_CHECK;
  cudaFree(d_pCoords);
  CUDA_CHECK;
  cudaFree(d_qCoords);
  CUDA_CHECK;
  cudaFree(d_imgOut);
  CUDA_CHECK;
}
