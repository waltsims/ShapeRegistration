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

#ifndef SHAPEREGISTRATION_H
#define SHAPEREGISTRATION_H

#include "helper.h"

#include <opencv2/opencv.hpp>
#include <vector>

#define BACKGROUND 0
#define FOREGROUND 1

using namespace std;
using namespace cv;

/** structure to save quad coordinates of each pixel
 *
 * \note need to check the order of the array index
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
/** setter for quad coordinates of each pixel
 *  \param[in] qCoords     array of quadCoords struct
 *  \param[in] w           size of width of the image
 *  \param[in] h           size of height of the image
 *  \param[in] xCoord      a x-coordinate of a pixel
 *  \param[in] yCoord      a y-coordinate of a pixel
 *  \param[out] qCoords    quad coordinates of each pixel
 *
 *  \retrun nothing
 */
void setQuadCoords (QuadCoords* qCoords, size_t w, size_t h);


/** cut margins of the image
 *  \param[in] imgIn              array of quadCoords struct
 *  \param[in] w                  size of width of the image
 *  \param[in] h                  size of height of the image
 *  \param[in, out] resizedImg    resized image which doesn't have margins
 *  \param[in, out] resizedW      size of width of the resized image
 *  \param[in, out] resizedH      size of height of the resized image
 *
 *  \retrun nothing
 */
void cutMargins (float* imgIn, size_t w, size_t h, float*& resizedImg, int& resizedW, int& resizedH);


/** set the center of mass of the image
 *  \param[in] imgIn              input image
 *  \param[in] w                  size of width of the image
 *  \param[in] h                  size of height of the image
 *  \param[in, out] xCentCoord    x-coordinte of the center of mass in the each channel
 *  \param[in, out] yCentCoord    y-coordinte of the center of mass in the each channel
 *
 *  \retrun nothing
 */
void centerOfMass (float *imgIn, size_t w, size_t h, float &xCentCoord, float &yCentCoord);



/** normalize the image coordinates
 *  \param[in] imgIn              input image
 *  \param[in] w                  size of width of the image
 *  \param[in] h                  size of height of the image
 *  \param[in, out] qCoords       quad coordinates of each pixel
 *  \param[in] xCentCoord         x-coordinte of the center of mass in the each channel
 *  \param[in] yCentCoord         y-coordinte of the center of mass in the each channel
 *
 *  \retrun nothing
 */
void imgNormalization (float *imgIn, size_t w, size_t h, QuadCoords* qCoords, float xCentCoord, float yCentCoord);

/** calculate moment of the image
 *  \param[in] imgIn         input image
 *  \param[in] w             size of width of the image
 *  \param[in] h             size of height of the image
 *  \param[in, out] mmt          an array for moments of the image
 *  \param[in, out] mmtDegree    the degree of moments
 *
 *  \retrun nothing
 *  \note pseudo code of geometric
 *moments(http://de.mathworks.com/matlabcentral/answers/71678-how-to-write-matlab-code-for-moments)
 */
void imageMoment(float *imgIn, size_t w, size_t h, float *mmt, size_t mmtDegree);


/** find vertices of polygon
 *  \param[in] affineParam        affine parameters
 *
 *  \retrun nothing
 *  \note http://stackoverflow.com/questions/33646643/store-details-of-a-binary-image-consisting-simple-polygons
 */
//void vertPoly(vector){

//}

/** check whether a point is inside polygon or not
 *  \param[in] nvert       Number of vertices in the polygon. Whether to repeat the first vertex at the end is discussed below.
 *  \param[in] vertx       Arrays containing the x-coordinates of the polygon's vertices.
 *  \param[in] verty       Arrays containing the y-coordinates of the polygon's vertices.
 *  \param[in] testx       X-coordinate of the test point.
 *  \param[in] testy       Y-coordinate of the test point.
 *
 *  \retrun                check result
 *  \note https://www.ecse.rpi.edu/Homepages/wrf/Research/Short_Notes/pnpoly.html#Polyhedron
 */
int pointInPolygon(int nVert, float *vertX, float *vertY, float testX, float testY);

/** thin plate spline
 *  \param[in] imgIn           input image
 *  \param[in] w               size of width of the image
 *  \param[in] h               size of height of the image
 *  \param[in, out] sigma      to be added
 *  \param[in] affineParam     affine parameters
 *  \param[in] ctrlP           c_k
 *  \param[in] localCoeff      local coefficients
 *
 *  \retrun nothing
 *  \note https://en.wikipedia.org/wiki/Thin_plate_spline
 */
void tps(float* imgIn, size_t w, size_t h, float *sigma, float *affineParam, float *ctrlP, float *localCoeff);

/** radial basis approximation
 *  \param[in] ctrlP         c_k
 *  \param[in] localP        x
 *  \param[in] localCoeff    w_ki
 *  \param[in] numP          index k
 *  \param[in] colInd        index i
 *  \param[out] sigma        to be added
 *
 *  \retrun give a new location
 *  \note https://en.wikipedia.org/wiki/Radial_basis_function
 */
float radialApprox(float *ctrlP, float localP, float *localCoeff, int numP, int i, int colInd);

#endif  // SHAPEREGISTRATION_H
