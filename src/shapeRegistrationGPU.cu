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

__global__ void imageMomentKernel(PixelCoords *d_pImg, int d_lenForeground, float *d_mmt,
                    int d_mmtDegree) {
  int x = threadIdx.x + blockDim.x * blockIdx.x;

  if (x < d_lenForeground) {
    // Compute all the combinations of the (p+q)-order image moments
    // Keep in mind that p,q go from 0 to mmtDegree-1.
    for (int q = 0; q < d_mmtDegree; q++) {
      for (int p = 0; p < d_mmtDegree; p++) {
        int mmtIndex = p + q * d_mmtDegree;

        // Compute the image moments taking the contributions from all the
        // pixels

        d_mmt[mmtIndex * d_lenForeground + x] = pow(d_pImg[x].x, p + 1) *
                                                pow(d_pImg[x].y, q + 1);
      }
    }
  }
}

void imageMomentGPU(PixelCoords *h_pImg, int h_lenForeground, float *h_mmt,
                    int h_mmtDegree) {

// void imageMomentGPU(float *h_imgIn, PixelCoords *h_pImg, int h_w, int h_h,
//                     float *h_mmt, int h_mmtDegree) {
  dim3 block = dim3(128, 1, 1);
  dim3 grid =
      dim3((h_lenForeground + block.x - 1) / block.x, 1, 1);

  PixelCoords *d_pImg;
  cudaMalloc(&d_pImg, h_lenForeground * sizeof(PixelCoords));
  CUDA_CHECK;

  float *d_mmt;
  cudaMalloc(&d_mmt, h_mmtDegree * h_mmtDegree * h_lenForeground * sizeof(float));
  CUDA_CHECK;

  // NOTE: check the size of array
  cudaMemcpy(d_pImg, h_pImg, h_lenForeground * sizeof(PixelCoords),
             cudaMemcpyHostToDevice);
  CUDA_CHECK;

  cudaMemset(d_mmt, 0, h_mmtDegree * h_mmtDegree * h_lenForeground * sizeof(float));
  CUDA_CHECK;
  
  imageMomentKernel << <grid, block>>>
      (d_pImg, h_lenForeground, d_mmt, h_mmtDegree);

  cudaMemcpy(h_mmt, d_mmt,
             h_mmtDegree * h_mmtDegree * h_lenForeground * sizeof(float),
             cudaMemcpyDeviceToHost);

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

// gpu version is much slower
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
  // gpu version is much slower
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

__global__ void pCoordsNormalisationKernel(int d_w, int d_h,
                                           PixelCoords *d_pCoords,
                                           float d_xCentCoord,
                                           float d_yCentCoord, float* d_normXFactor, float* d_normYFactor) {
  int index;

  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;

  *d_normXFactor = 0.5 / max(d_xCentCoord, d_w - d_xCentCoord);
  *d_normYFactor = 0.5 / max(d_yCentCoord, d_h - d_yCentCoord);

  if (x < d_w && y < d_h) {
    index = x + y * d_w;

    d_pCoords[index].x = (d_pCoords[index].x - d_xCentCoord) * *d_normXFactor;
    d_pCoords[index].y = (d_pCoords[index].y - d_yCentCoord) * *d_normYFactor;
  }
}

void pCoordsNormalisationGPU(int h_w, int h_h, PixelCoords *h_pCoords,
                             float h_xCentCoord, float h_yCentCoord, float &h_normXFactor, float &h_normYFactor) {
  dim3 block = dim3(16, 8, 1);
  dim3 grid =
      dim3((h_w + block.x - 1) / block.x, (h_h + block.y - 1) / block.y, 1);

  PixelCoords *d_pCoords;
  float* d_normXFactor;
  float* d_normYFactor;

  cudaMalloc(&d_pCoords, h_w * h_h * sizeof(PixelCoords));
  CUDA_CHECK;
  cudaMalloc(&d_normXFactor, sizeof(float));
  CUDA_CHECK;
  cudaMalloc(&d_normYFactor, sizeof(float));
  CUDA_CHECK;

  cudaMemcpy(d_pCoords, h_pCoords, h_w * h_h * sizeof(PixelCoords),
             cudaMemcpyHostToDevice);
  CUDA_CHECK;
  cudaMemset(d_normXFactor, 0, sizeof(float));
  CUDA_CHECK;
  cudaMemset(d_normYFactor, 0, sizeof(float));
  CUDA_CHECK;

  pCoordsNormalisationKernel << <grid, block>>>
      (h_w, h_h, d_pCoords, h_xCentCoord, h_yCentCoord, d_normXFactor, d_normYFactor);

  cudaMemcpy(h_pCoords, d_pCoords, h_w * h_h * sizeof(PixelCoords),
             cudaMemcpyDeviceToHost);
  CUDA_CHECK;
  cudaMemcpy(&h_normXFactor, d_normXFactor, sizeof(float),
             cudaMemcpyDeviceToHost);
  CUDA_CHECK;
  cudaMemcpy(&h_normYFactor, d_normYFactor, sizeof(float),
             cudaMemcpyDeviceToHost);
  CUDA_CHECK;

  cudaFree(d_pCoords);
  CUDA_CHECK;
  cudaFree(d_normXFactor);
  CUDA_CHECK;
  cudaFree(d_normYFactor);
  CUDA_CHECK;  
}

__global__ void qCoordsNormalisationKernel(int d_w, int d_h,
                                           QuadCoords *d_qCoords,
                                           float d_xCentCoord,
                                           float d_yCentCoord, float* d_normXFactor, float* d_normYFactor) {
  int index;

  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;

  *d_normXFactor = 0.5 / max(d_xCentCoord, d_w - d_xCentCoord);
  *d_normYFactor = 0.5 / max(d_yCentCoord, d_h - d_yCentCoord);

  if (x < d_w && y < d_h) {
    index = x + y * d_w;
    d_qCoords[index].x[0] =
        (d_qCoords[index].x[0] - d_xCentCoord) * *d_normXFactor;
    d_qCoords[index].y[0] =
        (d_qCoords[index].y[0] - d_yCentCoord) * *d_normYFactor;
    d_qCoords[index].x[1] =
        (d_qCoords[index].x[1] - d_xCentCoord) * *d_normXFactor;
    d_qCoords[index].y[1] =
        (d_qCoords[index].y[1] - d_yCentCoord) * *d_normYFactor;
    d_qCoords[index].x[2] =
        (d_qCoords[index].x[2] - d_xCentCoord) * *d_normXFactor;
    d_qCoords[index].y[2] =
        (d_qCoords[index].y[2] - d_yCentCoord) * *d_normYFactor;
    d_qCoords[index].x[3] =
        (d_qCoords[index].x[3] - d_xCentCoord) * *d_normXFactor;
    d_qCoords[index].y[3] =
        (d_qCoords[index].y[3] - d_yCentCoord) * *d_normYFactor;
  }
}

void qCoordsNormalisationGPU(int h_w, int h_h, QuadCoords *h_qCoords,
                             float h_xCentCoord, float h_yCentCoord, float &h_normXFactor, float &h_normYFactor) {
  dim3 block = dim3(16, 8, 1);
  dim3 grid =
      dim3((h_w + block.x - 1) / block.x, (h_h + block.y - 1) / block.y, 1);

  QuadCoords *d_qCoords;
  float* d_normXFactor;
  float* d_normYFactor;

  cudaMalloc(&d_qCoords, h_w * h_h * sizeof(QuadCoords));
  CUDA_CHECK;
  cudaMalloc(&d_normXFactor, sizeof(float));
  CUDA_CHECK;
  cudaMalloc(&d_normYFactor, sizeof(float));
  CUDA_CHECK;

  cudaMemcpy(d_qCoords, h_qCoords, h_w * h_h * sizeof(QuadCoords),
             cudaMemcpyHostToDevice);
  CUDA_CHECK;
  cudaMemset(d_normXFactor, 0, sizeof(float));
  CUDA_CHECK;
  cudaMemset(d_normYFactor, 0, sizeof(float));
  CUDA_CHECK;

  qCoordsNormalisationKernel << <grid, block>>>
      (h_w, h_h, d_qCoords, h_xCentCoord, h_yCentCoord, d_normXFactor, d_normYFactor);

  cudaMemcpy(h_qCoords, d_qCoords, h_w * h_h * sizeof(QuadCoords),
             cudaMemcpyDeviceToHost);
  CUDA_CHECK;
  cudaMemcpy(&h_normXFactor, d_normXFactor, sizeof(float),
             cudaMemcpyDeviceToHost);
  CUDA_CHECK;
  cudaMemcpy(&h_normYFactor, d_normYFactor, sizeof(float),
             cudaMemcpyDeviceToHost);
  CUDA_CHECK;

  cudaFree(d_qCoords);
  CUDA_CHECK;
  cudaFree(d_normXFactor);
  CUDA_CHECK;
  cudaFree(d_normYFactor);
  CUDA_CHECK;
}

__global__ void pCoordsDenormalisationKernel(int d_w, int d_h, PixelCoords *d_pCoords,
                            float d_xCentCoord, float d_yCentCoord) {

  int index;

  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;

  float normXFactor = 0.5 / max(d_xCentCoord, d_w - d_xCentCoord);
  float normYFactor = 0.5 / max(d_yCentCoord, d_h - d_yCentCoord);

  if(x < d_w && y < d_h) {
      index = x + (d_w * y);
      d_pCoords[index].x = (d_pCoords[index].x / normXFactor) + d_xCentCoord;
      d_pCoords[index].y = (d_pCoords[index].y / normYFactor) + d_yCentCoord;
  }

}

void pCoordsDenormalisationGPU(int h_w, int h_h, PixelCoords *h_pCoords,
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

  pCoordsDenormalisationKernel << <grid, block>>>
      (h_w, h_h, d_pCoords, h_xCentCoord, h_yCentCoord);

  cudaMemcpy(h_pCoords, d_pCoords, h_w * h_h * sizeof(PixelCoords),
             cudaMemcpyDeviceToHost);
  CUDA_CHECK;

  cudaFree(d_pCoords);
  CUDA_CHECK;
}

__device__ float radialApproxKernel(float d_x, float d_y, float d_cx,
                                    float d_cy) {
  float d_r2 = (d_cx - d_x) * (d_cx - d_x) + (d_cy - d_y) * (d_cy - d_y);

  return d_r2 < 0.0000000001 ? 0 : d_r2 * log(d_r2);
}

__global__ void pTPSGPUKernel(int d_lenForeground, PixelCoords *d_pCoords, TPSParams d_tpsParams,
             int d_cDim) {

  int x = threadIdx.x + blockDim.x * blockIdx.x;

  int dimSize = d_cDim * d_cDim;
  float Q;
  float freeDeformation[2] = {0, 0};

  if (x < d_lenForeground) {
    Q = 0;
    freeDeformation[0] = 0;
    freeDeformation[1] = 0;
    // for all c_m support coordinates
    for (int k = 0; k < dimSize; k++) {
      // calculate radial approximation

      Q = radialApproxKernel(d_pCoords[x].x, d_pCoords[x].y,
                             d_tpsParams.ctrlP[k],
                             d_tpsParams.ctrlP[k + dimSize]);

      // multiply with weights
      for (int i = 0; i < 2; i++) {
        freeDeformation[i] += d_tpsParams.localCoeff[k + (i * dimSize)] * Q;
      }
    }

    // note:: change
    float tempQCoordsX = d_pCoords[x].x;
    float tempQCoordsY = d_pCoords[x].y;

    d_pCoords[x].x = (d_tpsParams.affineParam[0] * tempQCoordsX) +
                         (d_tpsParams.affineParam[1] * tempQCoordsY) +
                         d_tpsParams.affineParam[2] + freeDeformation[0];

    d_pCoords[x].y = (d_tpsParams.affineParam[3] * tempQCoordsX) +
                         (d_tpsParams.affineParam[4] * tempQCoordsY) +
                         d_tpsParams.affineParam[5] + freeDeformation[1];
  }
}

void pTPSGPU(int h_lenForeground, PixelCoords *h_pCoords, TPSParams h_tpsParams,
             int h_cDim) {
  dim3 block = dim3(128, 1, 1);
  dim3 grid =
      dim3((h_lenForeground + block.x - 1) / block.x, 1, 1);

  PixelCoords *d_pCoords;
  cudaMalloc(&d_pCoords, h_lenForeground * sizeof(PixelCoords));
  CUDA_CHECK;

  cudaMemcpy(d_pCoords, h_pCoords, h_lenForeground * sizeof(PixelCoords),
             cudaMemcpyHostToDevice);
  CUDA_CHECK;

  pTPSGPUKernel <<<grid, block>>> (h_lenForeground, d_pCoords, h_tpsParams, h_cDim);
  CUDA_CHECK;

  cudaMemcpy(h_pCoords, d_pCoords, h_lenForeground * sizeof(PixelCoords),
             cudaMemcpyDeviceToHost);
  CUDA_CHECK;

  cudaFree(d_pCoords);
  CUDA_CHECK;
}

__global__ void qfTPSKernel(int d_lenForeground, QuadCoords *d_qCoords,
                           TPSParams d_tpsParams, int d_cDim) {

  int x = threadIdx.x + blockDim.x * blockIdx.x;

  int dimSize = d_cDim * d_cDim;
  float Q;
  float freeDeformation[2] = {0, 0};

  if (x < d_lenForeground) {
    for (int qIndex = 0; qIndex < 4; qIndex++) {
      Q = 0;
      freeDeformation[0] = 0;
      freeDeformation[1] = 0;
      // for all c_m support coordinates
      for (int k = 0; k < dimSize; k++) {
        // calculate radial approximation

        Q = radialApproxKernel(d_qCoords[x].x[qIndex],
                               d_qCoords[x].y[qIndex], d_tpsParams.ctrlP[k],
                               d_tpsParams.ctrlP[k + dimSize]);

        // multiply with weights
        for (int i = 0; i < 2; i++) {
          freeDeformation[i] += d_tpsParams.localCoeff[k + (i * dimSize)] * Q;
        }
      }

      // note:: change
      float tempQCoordsX = d_qCoords[x].x[qIndex];
      float tempQCoordsY = d_qCoords[x].y[qIndex];

      d_qCoords[x].x[qIndex] = (d_tpsParams.affineParam[0] * tempQCoordsX) +
                                   (d_tpsParams.affineParam[1] * tempQCoordsY) +
                                   d_tpsParams.affineParam[2] +
                                   freeDeformation[0];

      d_qCoords[x].y[qIndex] = (d_tpsParams.affineParam[3] * tempQCoordsX) +
                                   (d_tpsParams.affineParam[4] * tempQCoordsY) +
                                   d_tpsParams.affineParam[5] +
                                   freeDeformation[1];
    }
  }
}

void qfTPSGPU(int h_lenForeground, QuadCoords *h_qCoords, TPSParams &h_tpsParams,
             int h_cDim) {
  dim3 block = dim3(128, 1, 1);
  dim3 grid =
      dim3((h_lenForeground + block.x - 1) / block.x, 1, 1);

  QuadCoords *d_qCoords;
  cudaMalloc(&d_qCoords, h_lenForeground * sizeof(QuadCoords));
  CUDA_CHECK;

  cudaMemcpy(d_qCoords, h_qCoords, h_lenForeground * sizeof(QuadCoords),
             cudaMemcpyHostToDevice);
  CUDA_CHECK;

  qfTPSKernel << <grid, block>>> (h_lenForeground, d_qCoords, h_tpsParams, h_cDim);
  CUDA_CHECK;

  cudaMemcpy(h_qCoords, d_qCoords, h_lenForeground * sizeof(QuadCoords),
             cudaMemcpyDeviceToHost);
  CUDA_CHECK;

  cudaFree(d_qCoords);
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

__global__ void jacobianTransGPUKernel(int d_lenForeground, float *d_jacobi, PixelCoords * d_pCoords,
                                       TPSParams d_tpsParams, int d_c_dim) {
  int x = threadIdx.x + blockDim.x * blockIdx.x;

  int indexJ;
  int K = d_c_dim * d_c_dim;
  float squareOfNorm;
  float precomp;
  float x_j;
  float jacEl[4];

  if (x < d_lenForeground) {

    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        indexJ = i + 2 * j;
        jacEl[indexJ] = d_tpsParams.affineParam[i + 3 * j];
      }
    }

    __syncthreads();

    for (int k = 0; k < K; k++) {
      squareOfNorm =
          (d_tpsParams.ctrlP[k] - d_pCoords[x].x) * (d_tpsParams.ctrlP[k] - d_pCoords[x].x) +
          (d_tpsParams.ctrlP[k + K] - d_pCoords[x].y) * (d_tpsParams.ctrlP[k + K] - d_pCoords[x].y);

      if (squareOfNorm > 0.000001) {
        precomp = 2 * (1 + log(squareOfNorm));
      } else {
        precomp = 2;
      }

      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
          indexJ = i + 2 * j;
          x_j = (j == 0 ? d_pCoords[x].x : d_pCoords[x].y);
          jacEl[indexJ] -= precomp * d_tpsParams.localCoeff[k + i * K] *
                              (d_tpsParams.ctrlP[k + j * K] - x_j);
        }
      }
    }

    d_jacobi[x] = jacEl[0]*jacEl[3] - jacEl[1]*jacEl[2];
  }
}

// void jacobianTrans(int w, int h, float *jacobi, PixelCoords * pCoords,
//                    TPSParams &tpsParams, int c_dim)

void jacobianTransGPU(int h_lenForeground, float *h_jacobi, PixelCoords * h_pCoords,
                      TPSParams h_tpsParams, int h_c_dim) {
  dim3 block = dim3(128, 1, 1);
  dim3 grid =
      dim3((h_lenForeground + block.x - 1) / block.x, 1, 1);

  float *d_jacobi;
  PixelCoords *d_pCoords;

  cudaMalloc(&d_jacobi, h_lenForeground * sizeof(float));
  CUDA_CHECK;
  cudaMalloc(&d_pCoords, h_lenForeground * sizeof(PixelCoords));
  CUDA_CHECK;

  cudaMemset(d_jacobi, 0, h_lenForeground * sizeof(float));
  CUDA_CHECK;
  cudaMemcpy(d_pCoords, h_pCoords, h_lenForeground * sizeof(PixelCoords), cudaMemcpyHostToDevice);
  CUDA_CHECK;

  jacobianTransGPUKernel << <grid, block>>>
      (h_lenForeground, d_jacobi, d_pCoords, h_tpsParams, h_c_dim);
  CUDA_CHECK;

  cudaMemcpy(h_jacobi, d_jacobi, h_lenForeground * sizeof(float),
             cudaMemcpyDeviceToHost);
  CUDA_CHECK;
  cudaMemcpy(h_pCoords, d_pCoords, h_lenForeground * sizeof(PixelCoords),
             cudaMemcpyDeviceToHost);
  CUDA_CHECK;

  cudaFree(d_jacobi);
  CUDA_CHECK;
  cudaFree(d_pCoords);
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
  if (i == 0 && j == 0) {
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

void lmminObjectiveWrapperGPU(const double *par, const int m_dat,
                              const void *data, double *residual,
                              int *userbreak) {
  // The affineParam and the localCoeff are our free variables ("parameters")
  // and
  // need to be packed in an array in order to use the lmmin(). We pass
  // them as *par, but our functions are implemented to use the TPSParams
  // structure. We do the unpacking here.
  TPSParams tpsParams;

  for (int i = 0; i < 6; i++) {
    tpsParams.affineParam[i] = par[i];
  }

  // printf("affineParam[0] = %f, [1] = %f, [2] = %f\n",
  // tpsParams.affineParam[0], tpsParams.affineParam[1],
  // tpsParams.affineParam[2]);
  // printf("affineParam[3] = %f, [4] = %f, [5] = %f\n",
  // tpsParams.affineParam[3], tpsParams.affineParam[4],
  // tpsParams.affineParam[5]);

  for (int i = 0; i < 2 * DIM_C_REF * DIM_C_REF; i++) {
    tpsParams.localCoeff[i] = par[i + 6];
    // printf("localCoeff[i] = %f\n", tpsParams.localCoeff[i]);
  }

  // printf("tpsParams affine first: %f, last: %f\n", tpsParams.affineParam[0],
  // tpsParams.affineParam[5]);
  // printf("tpsParams localC first: %f, last: %f\n", tpsParams.localCoeff[0],
  // tpsParams.localCoeff[2 * DIM_C_REF * DIM_C_REF - 1]);

  // Cast the void pointer data to a float pointer dataF
  const float *dataF = static_cast<const float *>(data);

  // We also need to pack/unpack the non-free parameters ("data") of the
  // objective function
  // current reading position in the data array
  int offset = 0;

  // Read first the sizes needed to allocate the included arrays
  // int rt_w = static_cast<int>(data[offset]);
  int rt_w = dataF[offset];
  int rt_h = dataF[offset + 1];
  int ro_w = dataF[offset + 2];
  int ro_h = dataF[offset + 3];
  // We read 4 elements, move the reading position 4 places
  offset += 4;

  // printf("rt_w = %d, rt_h = %d, ro_w = %d, ro_h = %d\n", rt_w, rt_h, ro_w,
  // ro_h);

  // Template image array
  float *templateImg = new float[rt_w * rt_h];
  for (int i = 0; i < rt_w * rt_h; i++) {
    templateImg[i] = dataF[offset + i];
  }
  offset += rt_w * rt_h;

  // printf("templateImg first = %f, last = %f\n", templateImg[0],
  // templateImg[rt_w * rt_h - 1]);

  // Observation image array
  float *observationImg = new float[ro_w * ro_h];
  for (int i = 0; i < ro_w * ro_h; i++) {
    observationImg[i] = dataF[offset + i];
  }
  offset += ro_w * ro_h;

  // printf("observationImg first = %f, last = %f\n", observationImg[0],
  // observationImg[rt_w * rt_h - 1]);

  // Normalisation factors (N_i for eq.22)
  double Normalisation[81];  // TODO: Make this double everywhere
  for (int i = 0; i < 81; i++) {
    Normalisation[i] = dataF[offset + i];
  }
  offset += 81;

  // printf("Normalisation first = %f, last = %f\n", Normalisation[0],
  // Normalisation[80]);

  // Pixel coordinates of the template
  // Every element is a struct with two fields: x, y
  PixelCoords *pTemplate = new PixelCoords[rt_w * rt_h];
  for (int i = 0; i < rt_w * rt_h; i++) {
    pTemplate[i].x = dataF[offset + 2 * i];
    pTemplate[i].y = dataF[offset + 2 * i + 1];
  }
  offset += 2 * rt_w * rt_h;

  // printf("pTemplate first.x = %f, first.y = %f, last.x = %f, last.y = %f\n",
  // pTemplate[0].x, pTemplate[0].y, pTemplate[rt_w * rt_h-1].x, pTemplate[rt_w
  // * rt_h-1].y);

  // Quad coordinates of the template
  // Every element has two fields (x,y) that are arrays of four elements
  // (corners)
  QuadCoords *qTemplate = new QuadCoords[rt_w * rt_h];
  for (int i = 0; i < rt_w * rt_h; i++) {
    qTemplate[i].x[0] = dataF[offset + 8 * i];
    qTemplate[i].y[0] = dataF[offset + 8 * i + 1];
    qTemplate[i].x[1] = dataF[offset + 8 * i + 2];
    qTemplate[i].y[1] = dataF[offset + 8 * i + 3];
    qTemplate[i].x[2] = dataF[offset + 8 * i + 4];
    qTemplate[i].y[2] = dataF[offset + 8 * i + 5];
    qTemplate[i].x[3] = dataF[offset + 8 * i + 6];
    qTemplate[i].y[3] = dataF[offset + 8 * i + 7];
  }
  offset += 8 * rt_w * rt_h;

  // printf("qTemplate first.x[0] = %f, first.y[3] = %f, last.x[0] = %f,
  // last.y[3] = %f\n", qTemplate[0].x[0], qTemplate[0].y[3], qTemplate[rt_w *
  // rt_h-1].x[0], qTemplate[rt_w * rt_h-1].y[3]);

  // Pixel coordinates of the observation
  // Every element is a struct with two fields: x, y
  PixelCoords *pObservation = new PixelCoords[ro_w * ro_h];
  for (int i = 0; i < ro_w * ro_h; i++) {
    pObservation[i].x = dataF[offset + 2 * i];
    pObservation[i].y = dataF[offset + 2 * i + 1];
  }
  offset += 2 * ro_w * ro_h;

  // Normalisation factors of the template
  float t_sx, t_sy;
  t_sx = dataF[offset];
  t_sy = dataF[offset + 1];
  offset += 2;

  // Normalisation factors of the observation
  float o_sx, o_sy;
  o_sx = dataF[offset];
  o_sy = dataF[offset + 1];
  offset += 2;

  // printf("pObservation first.x = %f, last.y = %f\n", pObservation[0].x,
  // pObservation[ro_w * ro_h -1].y);

  // Array of the residuals of the equations
  // TODO: Add also the 6 extra equations!
  // printf("residual first = %f, last = %f\n", residual[0], residual[80]);

  // Call the objective function with the unpacked arguments
  objectiveFunctionGPU(observationImg, templateImg, ro_w, ro_h, Normalisation,
                    tpsParams, qTemplate, pTemplate, pObservation, rt_w, rt_h,
                    t_sx, t_sy, o_sx, o_sy, residual);

  // printf("residual first = %f, last = %f\n", residual[0], residual[80]);

  // Delete the allocated pointers
  delete templateImg;
  delete observationImg;
  delete pTemplate;
  delete qTemplate;

  return;
}

void objectiveFunctionGPU(float *observationImg, float *templateImg, int ro_w,
                          int ro_h, double *normalisation, TPSParams &tpsParams,
                          QuadCoords *qTemplate, PixelCoords *pTemplate,
                          PixelCoords *pObservation, int rt_w, int rt_h,
                          float t_sx, float t_sy, float o_sx, float o_sy,
                          double *residual) {
  static unsigned int call_count = 0;
  printf("call count = %d\n", call_count++);

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
  int o_lenForeground;
  int t_lenForeground;

  // get the size of the foreground array
  o_lenForeground = getNumForeground(observationImg , ro_w, ro_h) ;
  t_lenForeground = getNumForeground(templateImg, rt_w, rt_h);

  //create foreground array
  PixelCoords * pfObservation = new PixelCoords[o_lenForeground];
  PixelCoords * pfTemplate = new PixelCoords[t_lenForeground];
  
  // create image moment array
  float * observationMoment = new float[momentDeg * momentDeg * o_lenForeground];
  float * templateMoment= new float[momentDeg * momentDeg * t_lenForeground];

  //get coordinates of foreground
  getCoordForeground(observationImg, pObservation,ro_w, ro_h,
                        pfObservation) ;
  getCoordForeground(templateImg, pTemplate, rt_w,  rt_h,
                        pfTemplate) ;
  
  // get the jacobian at each pixel with the current tps params
  float jacobi[t_lenForeground];
  jacobianTransGPU(t_lenForeground, jacobi, pfTemplate, tpsParams, DIM_C_REF);

  // calculate tps transformation of template
  pTPSGPU(t_lenForeground, pfTemplate, tpsParams, DIM_C_REF);

  // get the moments of the TPS transformation of the template
  imageMomentGPU(pfTemplate, t_lenForeground, templateMoment, momentDeg);
  // get the moments of the observation
  imageMomentGPU(pfObservation, o_lenForeground, observationMoment, momentDeg);

  //fast clean up
  delete[] pfObservation;
  delete[] pfTemplate;

  // Determinant of the normFactor of the normalized template image
  float detN1 = 0;
  detN1 = t_sx * t_sy;
  // Determinant of the normFactor of the normalized observation image
  float detN2 = 0;
  detN2 = o_sx * o_sy;

  // Sum the moments of each degree for each pixel of the two images
  for (int index = 0; index < momentDeg * momentDeg; index++) {
    // Transformed template
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

  // // First restriction of eq.16 (2 equations)
  // int index = momentDeg * momentDeg;
  // residual[index] = 0;
  // residual[index+1] = 0;
  // int K = DIM_C_REF*DIM_C_REF;
  // for (int k = 0; k < K; k++) {
  //   residual[index]   += tpsParams.localCoeff[k];
  //   residual[index+1] += tpsParams.localCoeff[k + K];
  // }
  // resNorm += residual[index] * residual[index];
  // resNorm += residual[index+1] * residual[index+1];

  // index += 2;
  // // Second restriction of eq.16 (4 equations)
  // for (int i = 0; i < 2; i++) {
  //   for (int j = 0; j < 2; j++) {
  //     residual[index + (i + 2*j)] = 0;
  //     for (int k = 0; k < K; k++) {
  //       residual[index + (i + 2*j)] += tpsParams.ctrlP[k + j*K] * tpsParams.localCoeff[k + i*K];
  //     }
  //     resNorm += residual[index + (i + 2*j)] * residual[index + (i + 2*j)];
  //   }
  // }

  // Print the residual norm
  resNorm = sqrt(resNorm);
  printf("Residual norm = %f\n", resNorm);

  delete[] observationMoment;
  delete[] templateMoment;
};