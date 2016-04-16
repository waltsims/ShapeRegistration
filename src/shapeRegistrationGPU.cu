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
  dim3 block = dim3(16, 8, 1);
  dim3 grid =
      dim3((h_w + block.x - 1) / block.x, (h_h + block.y - 1) / block.y, 1);

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
  dim3 block = dim3(16, 8, 1);
  dim3 grid =
      dim3((h_w + block.x - 1) / block.x, (h_h + block.y - 1) / block.y, 1);

  PixelCoords *d_pImg;
  cudaMalloc(&d_pImg, h_w * h_h * sizeof(PixelCoords));
  CUDA_CHECK;

  float *d_imgIn;
  cudaMalloc(&d_imgIn, h_w * h_h * sizeof(float));
  CUDA_CHECK;

  float *d_mmt;
  cudaMalloc(&d_mmt, h_mmtDegree * h_mmtDegree * h_w * h_h * sizeof(float));
  CUDA_CHECK;

  cudaMemcpy(d_imgIn, h_imgIn, h_w * h_h * sizeof(float),
             cudaMemcpyHostToDevice);
  CUDA_CHECK;
  cudaMemcpy(d_pImg, h_pImg, h_w * h_h * sizeof(PixelCoords),
             cudaMemcpyHostToDevice);
  CUDA_CHECK;

  imageMomentKernel << <grid, block>>>
      (d_imgIn, d_pImg, h_w, h_h, d_mmt, h_mmtDegree);

  cudaMemcpy(h_mmt, d_mmt,
             h_mmtDegree * h_mmtDegree * h_w * h_h * sizeof(float),
             cudaMemcpyDeviceToHost);

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
  dim3 block = dim3(16, 8, 1);
  dim3 grid =
      dim3((h_w + block.x - 1) / block.x, (h_h + block.y - 1) / block.y, 1);

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

//gpu version is much slower
void cutMargins(float *imgIn, int w, int h, int &resizedW, int &resizedH,
                Margins &margins) {
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
}

__global__ void cutMarginsGPUKernel(float *d_imgIn, float *d_resizedImg,
                                    int d_w, int d_resizedW, int d_resizedH,
                                    Margins d_margins) {
  int resizedImgIndex;
  int originImgIndex;

  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;

  if (x < d_resizedW && y < d_resizedH) {
    resizedImgIndex = x + y * d_resizedW;
    originImgIndex = x + d_margins.left + (d_w * (y + d_margins.top));

    d_resizedImg[resizedImgIndex] = d_imgIn[originImgIndex];
  }
}

void cutMarginsGPU(float *h_imgIn, int h_w, int h_h, float *&h_resizedImg,
                   int &h_resizedW, int &h_resizedH, Margins &h_margins) {

  //gpu version is much slower
  cutMargins(h_imgIn, h_w, h_h, h_resizedW, h_resizedH, h_margins);

  h_resizedImg = new float[h_resizedW * h_resizedH];

  dim3 block = dim3(h_resizedW, 1, 1);
  dim3 grid = dim3((h_resizedW + block.x - 1) / block.x,
                   (h_resizedH + block.y - 1) / block.y, 1);

  float *d_imgIn;
  float *d_resizedImg;

  cudaMalloc(&d_imgIn, h_w * h_h * sizeof(float));
  CUDA_CHECK;

  cudaMalloc(&d_resizedImg, h_resizedW * h_resizedH * sizeof(float));
  CUDA_CHECK;

  cudaMemcpy(d_imgIn, h_imgIn, h_w * h_h * sizeof(float),
             cudaMemcpyHostToDevice);

  cudaMemset(d_resizedImg, 0, h_resizedW * h_resizedH * sizeof(float));
  CUDA_CHECK;

  cutMarginsGPUKernel << <grid, block>>>
      (d_imgIn, d_resizedImg, h_w, h_resizedW, h_resizedH, h_margins);
  CUDA_CHECK;

  cudaMemcpy(h_resizedImg, d_resizedImg,
             h_resizedW * h_resizedH * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_imgIn);
  CUDA_CHECK;
  cudaFree(d_resizedImg);
  CUDA_CHECK;
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
                                   float *d_xCentCoord, float *d_yCentCoord,
                                   float *numberOfForegroundPixel) {
  int index;

  int x_t = threadIdx.x;
  int y_d = blockIdx.x;

  extern __shared__ float s_data[];

  if (x_t < d_w && y_d < d_h) {
    s_data[x_t] = 0.0;
    s_data[x_t + d_w] = 0.0;
    s_data[x_t + d_w + d_w] = 0.0;
  }

  __syncthreads();

  if (x_t < d_w && y_d < d_h) {
    index = x_t + d_w * y_d;
    if (d_imgIn[index] == FOREGROUND) {
      atomicAdd(&s_data[x_t], float(x_t));
      atomicAdd(&s_data[x_t + d_w], float(y_d));
      atomicAdd(&s_data[x_t + d_w + d_w], 1.0f);
    }
  }

  __syncthreads();
  if (x_t == 0 && y_d == 1) {
  }
  if (x_t < d_w && y_d < d_h) {
    for (int offset = d_w / 2; offset > 0; offset /= 2) {
      if (x_t < offset) {
        s_data[x_t] += s_data[x_t + offset];
        s_data[x_t + d_w] += s_data[x_t + d_w + offset];
        s_data[x_t + d_w + d_w] += s_data[x_t + d_w + d_w + offset];
      }
      __syncthreads();
    }
  }

  __syncthreads();

  if (x_t == 0 && y_d < d_h) {
    d_xCentCoord[y_d] = s_data[0];
    d_yCentCoord[y_d] = s_data[d_w];
    numberOfForegroundPixel[y_d] = s_data[d_w + d_w];
  }
}

void centerOfMassGPU(float *h_imgIn, int h_w, int h_h, float &h_xCentCoord,
                     float &h_yCentCoord) {
  const int nThreads = h_w;

  printf("h_w, h_h : (%d, %d)\n", h_w, h_h);

  dim3 block = dim3(nThreads, 1, 1);
  dim3 grid = dim3(h_w, 1, 1);

  float *numOfForegroundPixelOut = new float[h_w * sizeof(float)];
  float *xCentCoordsOut = new float[h_w * sizeof(float)];
  float *yCentCoordsOut = new float[h_w * sizeof(float)];

  float *d_imgIn;
  float *d_xCentCoord;
  float *d_yCentCoord;
  float *d_numOfForegroundPixelOut;

  cudaMalloc(&d_imgIn, h_w * h_h * sizeof(float));
  CUDA_CHECK;
  cudaMalloc(&d_xCentCoord, h_w * sizeof(float));
  CUDA_CHECK;
  cudaMalloc(&d_yCentCoord, h_h * sizeof(float));
  CUDA_CHECK;
  cudaMalloc(&d_numOfForegroundPixelOut, h_h * sizeof(float));
  CUDA_CHECK;

  cudaMemcpy(d_imgIn, h_imgIn, h_w * h_h * sizeof(float),
             cudaMemcpyHostToDevice);
  CUDA_CHECK;
  cudaMemset(d_xCentCoord, 0, h_w * sizeof(float));
  CUDA_CHECK;
  cudaMemset(d_yCentCoord, 0, h_h * sizeof(float));
  CUDA_CHECK;
  cudaMemset(d_numOfForegroundPixelOut, 0, h_h * sizeof(float));
  CUDA_CHECK;

  centerOfMassKernel << <grid, block, 3 * nThreads * sizeof(float)>>>
      (d_imgIn, h_w, h_h, d_xCentCoord, d_yCentCoord,
       d_numOfForegroundPixelOut);
  CUDA_CHECK;

  cudaMemcpy(xCentCoordsOut, d_xCentCoord, h_w * sizeof(float),
             cudaMemcpyDeviceToHost);
  CUDA_CHECK;
  cudaMemcpy(yCentCoordsOut, d_yCentCoord, h_w * sizeof(float),
             cudaMemcpyDeviceToHost);
  CUDA_CHECK;
  cudaMemcpy(numOfForegroundPixelOut, d_numOfForegroundPixelOut,
             h_w * sizeof(float), cudaMemcpyDeviceToHost);
  CUDA_CHECK;

  h_xCentCoord = 0;
  h_yCentCoord = 0;
  for (int i = 0; i < h_w; i++) {
    h_xCentCoord += xCentCoordsOut[i];
    h_yCentCoord += yCentCoordsOut[i];
    if (i != 0) numOfForegroundPixelOut[0] += numOfForegroundPixelOut[i];
  }

  h_xCentCoord /= numOfForegroundPixelOut[0];
  h_yCentCoord /= numOfForegroundPixelOut[0];

  cudaFree(d_imgIn);
  CUDA_CHECK;
  cudaFree(d_xCentCoord);
  CUDA_CHECK;
  cudaFree(d_yCentCoord);
  CUDA_CHECK;
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
  dim3 block = dim3(16, 8, 1);
  dim3 grid =
      dim3((h_w + block.x - 1) / block.x, (h_h + block.y - 1) / block.y, 1);

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
  dim3 block = dim3(16, 8, 1);
  dim3 grid =
      dim3((h_w + block.x - 1) / block.x, (h_h + block.y - 1) / block.y, 1);

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

__device__ float radialApproxKernel(float d_x, float d_y, float d_cx,
                                    float d_cy) {
  float d_r2 = (d_cx - d_x) * (d_cx - d_x) + (d_cy - d_y) * (d_cy - d_y);

  return d_r2 < 0.0000000001 ? 0 : d_r2 * log(d_r2);
}

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

      Q = radialApproxKernel(d_pCoords[index].x, d_pCoords[index].y,
                             d_tpsParams.ctrlP[k],
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
                         d_tpsParams.affineParam[2] + freeDeformation[0];

    d_pCoords[index].y = (d_tpsParams.affineParam[3] * tempQCoordsX) +
                         (d_tpsParams.affineParam[4] * tempQCoordsY) +
                         d_tpsParams.affineParam[5] + freeDeformation[1];
  }
}

void pTPSGPU(int h_w, int h_h, PixelCoords *h_pCoords, TPSParams &h_tpsParams,
             int h_cDim) {
  dim3 block = dim3(16, 8, 1);
  dim3 grid =
      dim3((h_w + block.x - 1) / block.x, (h_h + block.y - 1) / block.y, 1);

  PixelCoords *d_pCoords;
  cudaMalloc(&d_pCoords, h_h * h_w * sizeof(PixelCoords));
  CUDA_CHECK;

  cudaMemcpy(d_pCoords, h_pCoords, h_h * h_w * sizeof(PixelCoords),
             cudaMemcpyHostToDevice);
  CUDA_CHECK;

  pTPSGPUKernel << <grid, block>>> (h_w, h_h, d_pCoords, h_tpsParams, h_cDim);
  CUDA_CHECK;

  cudaMemcpy(h_pCoords, d_pCoords, h_h * h_w * sizeof(PixelCoords),
             cudaMemcpyDeviceToHost);
  CUDA_CHECK;

  cudaFree(d_pCoords);
  CUDA_CHECK;
}

__global__ void qTPSKernel(int d_w, int d_h, QuadCoords *d_qCoords,
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
                                   d_tpsParams.affineParam[5] +
                                   freeDeformation[1];
    }
  }
}

void qTPSGPU(int h_w, int h_h, QuadCoords *h_qCoords, TPSParams &h_tpsParams,
             int h_cDim) {
  dim3 block = dim3(16, 8, 1);
  dim3 grid =
      dim3((h_w + block.x - 1) / block.x, (h_h + block.y - 1) / block.y, 1);

  QuadCoords *d_qCoords;
  cudaMalloc(&d_qCoords, h_h * h_w * sizeof(QuadCoords));
  CUDA_CHECK;

  cudaMemcpy(d_qCoords, h_qCoords, h_h * h_w * sizeof(QuadCoords),
             cudaMemcpyHostToDevice);
  CUDA_CHECK;

  qTPSKernel << <grid, block>>> (h_w, h_h, d_qCoords, h_tpsParams, h_cDim);
  CUDA_CHECK;

  cudaMemcpy(h_qCoords, d_qCoords, h_h * h_w * sizeof(QuadCoords),
             cudaMemcpyDeviceToHost);
  CUDA_CHECK;

  cudaFree(d_qCoords);
  CUDA_CHECK;
}

__global__ void jacobianTransGPUKernel(int d_w, int d_h, float *d_jacobi,
                                       TPSParams &d_tpsParams, int d_c_dim) {
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;

  int indexP, indexJ;
  int K = d_c_dim * d_c_dim;
  float squareOfNorm;
  float precomp;
  float x_j;

  if (x < d_w && y < d_h) {
    indexP = x + d_w * y;

    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        indexJ = 4 * indexP + i + 2 * j;
        d_jacobi[indexJ] = d_tpsParams.affineParam[i + 2 * j];
      }
    }

    __syncthreads();

    for (int k = 0; k < K; k++) {
      squareOfNorm =
          (d_tpsParams.ctrlP[k] - x) * (d_tpsParams.ctrlP[k] - x) +
          (d_tpsParams.ctrlP[k + K] - y) * (d_tpsParams.ctrlP[k + K] - y);

      if (squareOfNorm > 0.000001) {
        precomp = 2 * (1 + log(squareOfNorm));
      } else {
        precomp = 2;
      }

      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
          indexJ = 4 * indexP + i + 2 * j;
          x_j = (j == 0 ? x : y);
          d_jacobi[indexJ] -= precomp * d_tpsParams.localCoeff[k + i * K] *
                              (d_tpsParams.ctrlP[k + j * K] - x_j);
        }
      }
    }
  }
}

void jacobianTransGPU(int h_w, int h_h, float *h_jacobi, TPSParams h_tpsParams,
                      int h_c_dim) {
  dim3 block = dim3(16, 8, 1);
  dim3 grid =
      dim3((h_w + block.x - 1) / block.x, (h_h + block.y - 1) / block.y, 1);

  float *d_jacobi;
  cudaMalloc(&d_jacobi, h_w * h_h * 4 * sizeof(float));
  CUDA_CHECK;
  cudaMemset(d_jacobi, 0, h_w * h_h * 4 * sizeof(float));
  CUDA_CHECK;

  jacobianTransGPUKernel << <grid, block>>>
      (h_w, h_h, d_jacobi, h_tpsParams, h_c_dim);
  CUDA_CHECK;

  cudaMemcpy(h_jacobi, d_jacobi, h_h * h_w * 4 * sizeof(float),
             cudaMemcpyDeviceToHost);
  CUDA_CHECK;

  cudaFree(d_jacobi);
  CUDA_CHECK;
}

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
  if( i == 0 && j == 0) {
    printf("transferkernel : %d, %d\n", d_o_w, d_o_h);
  }

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

  printf("transfergpu : %d, %d\n", h_o_w, h_o_h);
  dim3 block = dim3(16, 8, 1);
  dim3 grid =
      dim3((h_t_w + block.x - 1) / block.x, (h_t_h + block.y - 1) / block.y, 1);

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
