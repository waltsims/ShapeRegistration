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

/** structure to save the middle coord of each pixel
 *
 */

struct PixelCoords {
  float x;
  float y;
};

/** structure to save margin sizes
 * 
 */
struct Margins {

  int top = 0;
  int bottom = 0;
  int left = 0;
  int right = 0;
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

  void setQuadCoords(QuadCoords *qCoords, size_t w, size_t h);

/** setter for normalized center coordinates of each pixel
 *  \param[in] qCoords     array of quadCoords struct
 *  \param[in] w           size of width of the image
 *  \param[in] h           size of height of the image
 *  \param[in] xCoord      a x-coordinate of a pixel
 *  \param[in] yCoord      a y-coordinate of a pixel
 *  \param[out] qCoords    quad coordinates of each pixel
 *
 *  \retrun nothing
 */

void setPixelCoords(PixelCoords *pCoords, size_t w, size_t h);

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

void cutMargins(float *imgIn, size_t w, size_t h, float *&resizedImg,
                int &resizedW, int &resizedH, Margins &margins);

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

void addMargins(float *resizedimg, int resizedW, int resizedH, float *imgOut,
                size_t w, size_t h, Margins &margins);

/** set the center of mass of the image
 *  \param[in] imgIn              input image
 *  \param[in] w                  size of width of the image
 *  \param[in] h                  size of height of the image
 *  \param[in, out] xCentCoord    x-coordinte of the center of mass in the each
 * channel
 *  \param[in, out] yCentCoord    y-coordinte of the center of mass in the each
 * channel
 *
 *  \retrun nothing
 */
void centerOfMass(float *imgIn, size_t w, size_t h, float &xCentCoord,
                  float &yCentCoord);

/** normalize the image coordinates
 *  \param[in] imgIn              input image
 *  \param[in] w                  size of width of the image
 *  \param[in] h                  size of height of the image
 *  \param[in, out] pCoords       center coordinates of each pixel
 *  \param[in] xCentCoord         x-coordinte of the center of mass in image
 *  \param[in] yCentCoord         y-coordinte of the center of mass in image
 * channel
 *
 *  \retrun nothing
 */
void imgNormalization(size_t w, size_t h, PixelCoords *pCoords,
                      float xCentCoord, float yCentCoord);

/** inverse normalization of the image coordinates
 *  \param[in] imgIn              input image
 *  \param[in] w                  size of width of the image
 *  \param[in] h                  size of height of the image
 *  \param[in, out] pCoords       center coordinates of each pixel
 *  \param[in] xCentCoord         x-coordinte of the center of mass in image
 *  \param[in] yCentCoord         y-coordinte of the center of mass in image
 * channel
 *
 *  \retrun nothing
 */
void imgDenormalization(size_t w, size_t h, PixelCoords *pCoords,
                      float xCentCoord, float yCentCoord);

/** calculate moment of the image
 *  \param[in] imgIn             input image
 *  \param[in] w                 size of width of the image
 *  \param[in] h                 size of height of the image
 *  \param[in, out] mmt          an array for moments of the image
 *  \param[in, out] mmtDegree    the degree of moments
 *
 *  \retrun nothing
 *  \note pseudo code of geometric
 *moments(http://de.mathworks.com/matlabcentral/answers/71678-how-to-write-matlab-code-for-moments)
 */
void imageMoment(float *imgIn, size_t w, size_t h, float *mmt,
                 size_t mmtDegree);

/** find vertices of polygon
 *  \param[in] affineParam        affine parameters
 *
 *  \retrun nothing
 *  \note
 * http://stackoverflow.com/questions/33646643/store-details-of-a-binary-image-consisting-simple-polygons
 */
// void vertPoly(vector){

//}

/** check whether a point is inside polygon or not
 *  \param[in] nvert       Number of vertices in the polygon. Whether to repeat
 * the first vertex at the end is discussed below.
 *  \param[in] vertx       Arrays containing the x-coordinates of the polygon's
 * vertices.
 *  \param[in] verty       Arrays containing the y-coordinates of the polygon's
 * vertices.
 *  \param[in] testx       X-coordinate of the test point.
 *  \param[in] testy       Y-coordinate of the test point.
 *
 *  \retrun                check result
 *  \note
 * https://www.ecse.rpi.edu/Homepages/wrf/Research/Short_Notes/pnpoly.html#Polyhedron
 */
int pointInPolygon(int nVert, float *vertX, float *vertY, float testX,
                   float testY);

/** thin plate spline
 *  \param[in] w                    size of width of the image
 *  \param[in] h                    size of height of the image
 *  \param[in, out] sigma           to be added
 *  \param[in, out] affineParma     to be added
 *  \param[in, out] localCoeff      to be added
 *  \param[in, out] ctrlP           to be added
 *
 *  \retrun                         nothing
 */
void updateTPSVariables(size_t w, size_t h, float *sigma, float *affineParam,
                        float *localCoeff, float *ctrlP);

/** thin plate spline
 *  \param[in] imgIn           input image
 *  \param[in] w               size of width of the image
 *  \param[in] h               size of height of the image
 *  \param[in, out] sigma      to be added
 *  \param[in] affineParam     affine parameters
 *  \param[in] localCoeff      local coefficients
 *  \param[in] ctrlP           c_k
 *  \param[in] mmt             an array for moments of the image
 *  \param[in] mmtDegree       the degree of moments
 *
 *  \retrun                    nothing
 *  \note https://en.wikipedia.org/wiki/Thin_plate_spline
 */
void tps(float *imgIn, size_t w, size_t h, float *sigma, float *affineParam,
         float *localCoeff, float *ctrlP, float *mmt, int mmtDegree);

/** radial basis approximation
 *  \param[in] w               size of width of the image
 *  \param[in] h               size of height of the image
 *  \param[in, out] sigma      to be added
 *  \param[in] ctrlP           c_k
 *  \param[in] pVector         x
 *  \param[in] mmt             an array for moments of the image
 *  \param[in] mmtDegree       the degree of moments
 *  \param[in] dimIndex        index of dimension
 *
 *  \retrun give a new location
 */
float radialApprox(size_t w, size_t h, float *sigma, float *localCoeff,
                   float *ctrlP, size_t pVector, float *mmt, int mmtDegree,
                   int dimIndex);

#endif  // SHAPEREGISTRATION_H
