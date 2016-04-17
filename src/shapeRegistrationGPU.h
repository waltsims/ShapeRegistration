/** \file shapeRegistration.h
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

#ifndef SHAPEREGISTRATIONGPU_H
#define SHAPEREGISTRATIONGPU_H

#include "helper.h"

#include <opencv2/opencv.hpp>
#include <vector>

#define BACKGROUND 0
#define FOREGROUND 1

#define DIM_C_REF 5



/** structure to save parameters for thin plate spline
 *
 *  \note all the values are given from tps.mat
 *
 */
struct TPSParams {
  /** affine parameters: a
   *  size: 2 X 3
   */
  float affineParam[2 * 3] = {180.750570973114,
                -16.6979777565757,
                              78.2371003511860,
                26.8971153700975,
                              230.948397768629,
                94.8716771317028};
  /** local coefficient: w
   *  size: 2 X degree of moment^2
   */
  float localCoeff[2 * DIM_C_REF * DIM_C_REF] = {
                43.4336100367918,
                -254.664478125986,
                164.260796260200,
                209.676277829085,
                -95.4890564425583,
                74.0543824971685,
                -28.7045465000123,
                24.3081842513946,
                12.0776119309275,
                -110.537185437210,
                -31.4277765118847,
                -12.6475496029205,
                -6.28490818780619,
                -17.9631170829272,
                -49.2124002833565,
                4.55138433535774,
                -17.8818978065982,
                9.33577702465229,
                -6.71131995853042,
                63.3149271119866,
                11.6553014436196,
                -96.3055081066777,
                159.231246073828,
                -77.2364773866945,
                29.1667360924734,
                -114.139336032721,
                137.564797025932,
                9.98432048986034,
                -79.6383549015170,
                -40.5321424871105,
                -170.186873110339,
                -6.97667497242567,
                5.22216368408785,
                3.13977210362321,
                113.586062434194,
                238.165062497888,
                49.8182500014783,
                5.69215060608789,
                -9.06319782846551,
                -3.95545094132852,
                158.306011440768,
                17.3093272722814,
                -1.07041222178313,
                -29.4179219657275,
                -193.750653161620,
                -204.879616315085,
                -45.7180638714529,
                14.9243166793925,
                -39.5796905616838,
                185.196153840653};
  /** conrol points: c
   *  size: 2 X degree of moment^2
   *
   *  25 total points odered x then y dimensions
   */
  float ctrlP[2 * DIM_C_REF * DIM_C_REF] = {
    //xx
      -0.333333333333333,
    -0.333333333333333, -0.333333333333333,
      -0.333333333333333, -0.333333333333333, -0.166666666666667,
      -0.166666666666667, -0.166666666666667, -0.166666666666667,
      -0.166666666666667, 0, 0, 0, 0, 0, 0.166666666666667, 0.166666666666667,
      0.166666666666667, 0.166666666666667, 0.166666666666667,
      0.333333333333333, 0.333333333333333, 0.333333333333333,
      0.333333333333333, 0.333333333333333,

    -0.333333333333333,
      -0.166666666666667, 0, 0.166666666666667, 0.333333333333333,
      -0.333333333333333, -0.166666666666667, 0, 0.166666666666667,
      0.333333333333333, -0.333333333333333, -0.166666666666667, 0,
      0.166666666666667, 0.333333333333333, -0.333333333333333,
      -0.166666666666667, 0, 0.166666666666667, 0.333333333333333,
      -0.333333333333333, -0.166666666666667, 0, 0.166666666666667,
      0.333333333333333};
};
/** structure to save quad coordinates of each pixel
 *
 *  \note need to check the order of the array index
 *
 * (x-0.5, y-0.5)   (x+0.5, y-0.5)
 *  (x[0], y[0])     (x[1], y[1])
 *             0-----0
 *             |  0  |
 *             |(x,y)|
 *             0-----0
 *  (x[3], y[3])     (x[2], y[2])
 * (x-0.5, y+0.5)   (x+0.5, y+0.5)
 *
 */
struct QuadCoords {
  float x[4];
  float y[4];
};

/** structure to save the middle coord of each pixel
 *
 */
struct PixelCoords {
  float x;
  float y;
};

/** structure to save margin positions
 * top: row (y) of the first foreground pixel from top.
 * bottom: row (y) of the last foreground pixel from top.
 * left: column (x) of the first foreground pixel from left.
 * right: column (x) of the last foreground pixel from left.
 *
 */
struct Margins {
  int top = 0;
  int bottom = 0;
  int left = 0;
  int right = 0;
};



using namespace std;
using namespace cv;

/** structure to save parameters for thin plate spline
 *
 *  \note all the values are given from tps.mat
 *
 */
void setPixelCoordsGPU(PixelCoords *pCoords, int w, int h);

/** setter for quad coordinates of each pixel
 *  \param[in, out] qCoords     array of quadCoords struct
 *  \param[in] w                width of the image
 *  \param[in] h                height of the image
 *
 *  \return nothing
 */
void setQuadCoordsGPU(QuadCoords *qCoords, int w, int h);

void cutMargins(float *imgIn, int w, int h, int &resizedW, int &resizedH,
                Margins &margins);

/** cut margins of the image
 *  \param[in] imgIn              array of the input image pixels
 *  \param[in] w                  width of the image
 *  \param[in] h                  height of the image
 *  \param[out] resizedImg        resized image which doesn't have margins
 *(cropped)
 *  \param[out] resizedW          width of the resized image
 *  \param[out] resizedH          height of the resized image
 *  \param[out] margins           margin positions (first/last foreground
 *indices)
 *
 *  \return nothing
 */
void cutMarginsGPU(float *imgIn, int w, int h, float *&resizedImg,
                   int &resizedW, int &resizedH, Margins &margins);

/** add the original margins to a cropped image
 *  \param[in] resizedImg         array of the cropped image pixels
 *  \param[in] resizedW           width of the resized image
 *  \param[in] resizedH           height of the resized image
 *  \param[out] imgOut            array of the output, full size image pixels
 *  \param[in] w                  width of the image
 *  \param[in] h                  height of the image
 *  \param[in] margins            margin positions (first/last foreground
 *indices)
 *
 *  \return nothing
 */
void addMarginsGPU(float *resizedimg, int resizedW, int resizedH, float *imgOut,
                   int w, int h, Margins &margins);

/** set the center of mass of the image
 *  \param[in] imgIn              input image
 *  \param[in] w                  width of the image
 *  \param[in] h                  height of the image
 *  \param[in, out] xCentCoord    x-coordinate of the center of mass
 *  \param[in, out] yCentCoord    y-coordinate of the center of mass
 *
 *  \return nothing
 */
void centerOfMassGPU(float *h_imgIn, int h_w, int h_h, float &h_xCentCoord,
                     float &h_yCentCoord);

/** normalize the center coordinates for each pixel
 *  \param[in] w                  width of the image
 *  \param[in] h                  height of the image
 *  \param[in, out] pCoords       center coordinates of each pixel
 *  \param[in] xCentCoord         x-coordinate of the center of mass in image
 *  \param[in] yCentCoord         y-coordinate of the center of mass in image
 *
 *  \return nothing
 */
void pCoordsNormalisationGPU(int w, int h, PixelCoords *pCoords,
                             float xCentCoord, float yCentCoord, float &normXFactor, float &normYFactor);



/** normalize the quad coordinates for each pixel
 *  \param[in] w                  width of the image
 *  \param[in] h                  height of the image
 *  \param[in, out] qCoords       quad coordinates of each pixel
 *  \param[in] xCentCoord         x-coordinate of the center of mass in image
 *  \param[in] yCentCoord         y-coordinate of the center of mass in image
 *
 *  \return nothing
 */
void qCoordsNormalisationGPU(int w, int h, QuadCoords *qCoords,
                             float xCentCoord, float yCentCoord, float &normXFactor, float &normYFactor);


/** inverse Normalisation of the image coordinates
 *  \param[in] w                  width of the image
 *  \param[in] h                  height of the image
 *  \param[in, out] pCoords       center coordinates of each pixel
 *  \param[in] xCentCoord         x-coordinate of the center of mass in image
 *  \param[in] yCentCoord         y-coordinate of the center of mass in image
 *
 *  \return nothing
 */

void pCoordsDenormalisationGPU(int w, int h, PixelCoords *pCoords,
                               float xCentCoord, float yCentCoord);

int getNumForeground(float *imgIn, int w, int h);

void getCoordForeground(float *imgIn, PixelCoords *pImgIn, int w, int h,
                        PixelCoords *pForeground);

/** calculate moment of the image
 *  \param[in] imgIn             input image
 *  \param[in] w                 width of the image
 *  \param[in] h                 height of the image
 *  \param[in, out] mmt          an array for moments of the image
 *  \param[in, out] mmtDegree    the degree of moments (p,q in [0, mmtDegree) )
 *
 *  \return nothing
 *  \note pseudo code of geometric
 * moments(http://de.mathworks.com/matlabcentral/answers/71678-how-to-write-matlab-code-for-moments)
 */
void imageMomentGPU(float *imgIn, PixelCoords *pImg, int w, int h, float *mmt,
                    int mmtDegree);

/** thin plate spline for pixel coordinates
 *  \param[in] imgIn           input image
 *  \param[in] w               width of the image
 *  \param[in] h               height of the image
 *  \param[in, out] sigma      to be added
 *  \param[in] tpsParams       parameters for thisn plate spline method
 *  \param[in] mmt             an array for moments of the image
 *  \param[in] mmtDegree       the degree of moments
 *
 *  \return                    nothing
 *  \note https://en.wikipedia.org/wiki/Thin_plate_spline
 */
void pTPSGPU(int w, int h, PixelCoords *pCoords, TPSParams &tpsParams,
             int mmtDegree);

/** radial basis approximation
 *  \param[in] x         x-coordinates of current pixel as
 *structure
 *  \param[in] y         y-coordinates of current pixel as
 *structure
 *  \param[in] cx               cx-coordinate
 *  \param[in] cy               y-coordinate
 *
 *  \return radial basis function value
 */
float radialApproxGPU(float x, float y, float cx, float cy);

/** thin plate spline for quad coordinates
 *  \param[in] imgIn           input image
 *  \param[in] w               width of the image
 *  \param[in] h               height of the image
 *  \param[in, out] sigma      to be added
 *  \param[in] tpsParams       parameters for thisn plate spline method
 *  \param[in] mmt             an array for moments of the image
 *  \param[in] mmtDegree       the degree of moments
 *
 *  \return                    nothing
 *  \note https://en.wikipedia.org/wiki/Thin_plate_spline
 */
void qTPSGPU(int w, int h, QuadCoords *qCoords, TPSParams &tpsParams,
             int mmtDegree);

/** jacobian transformation
 *  \param[in] w               width of the image
 *  \param[in] h               height of the image
 *  \param[in, out] jacobi     array of size 4*w*h storing the four jacobian
 *                             elements for each pixel.
 *  \param[in] tpsParams       parameters for thin plate spline method
 *  \param[in] c_dim           number of control points
 *
 *  \return                    nothing
 *  \note https://en.wikipedia.org/wiki/Thin_plate_spline
 */

void jacobianTransGPU(int w, int h, float *jacobi, PixelCoords * pCoords,
                   TPSParams tpsParams, int c_dim) ;
/** Discription to come
 *
 * */
void transferGPU(float *imgIn, PixelCoords *pCoords, QuadCoords *qCoords,
                 int t_w, int t_h, int o_w, int o_h, float *imgOut);

/** check whether a point is inside polygon or not
 *  \param[in] nVert       Number of vertices in the polygon. Whether to repeat
 * the first vertex at the end is discussed below.
 *  \param[in] vertX       Arrays containing the x-coordinates of the polygon's
 * vertices.
 *  \param[in] vertY       Arrays containing the y-coordinates of the polygon's
 * vertices.
 *  \param[in] testx       X-coordinate of the test point.
 *  \param[in] testy       Y-coordinate of the test point.
 *
 *  \return                check result
 *  \note
 * https://www.ecse.rpi.edu/Homepages/wrf/Research/Short_Notes/pnpoly.html#Polyhedron
 */
bool pointInPolygonGPU(int nVert, float *vertX, float *vertY, float testX,
                       float testY);

void lmminObjectiveWrapperGPU(const double *par, const int m_dat,
                              const void *data, double *residual,
                              int *userbreak);

void objectiveFunctionGPU(float *observationImg, float *templateImg, int ro_w,
                          int ro_h, double *normalisation, TPSParams &tpsParams,
                          QuadCoords *qTemplate, PixelCoords *pTemplate,
                          PixelCoords *pObservation, int rt_w, int rt_h,
                          float t_sx, float t_sy, float o_sx, float o_sy,
                          double *residual);

#endif  // SHAPEREGISTRATION_H
