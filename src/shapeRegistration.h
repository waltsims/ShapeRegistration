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

/** calculate moment of the image
 *  \param[in] imgIn         input image
 *  \param[in] w             size of width of the image
 *  \param[in] h             size of height of the image
 *  \param[in] nc            the number of channels of the image
 *  \param[out] mmt          an array for moments of the image
 *  \param[out] mmtDegree    the degree of moments
 *
 *  \retrun nothing
 *  \note pseudo code of geometric
 *moments(http://de.mathworks.com/matlabcentral/answers/71678-how-to-write-matlab-code-for-moments)
 */
void imageMoment(float *imgIn, size_t w, size_t h, size_t nc, float *mmt,
                 size_t mmtDegree);

/** thin plate spline
 *  \param[in] nvert       Number of vertices in the polygon. Whether to repeat the first vertex at the end is discussed below. 
 *  \param[in] vertx       Arrays containing the x-coordinates of the polygon's vertices.
 *  \param[in] verty       Arrays containing the y-coordinates of the polygon's vertices.
 *  \param[in] testx       X-coordinate of the test point.
 *  \param[in] testy       Y-coordinate of the test point.
 *
 *  \retrun nothing
 *  \note http://alienryderflex.com/polygon/
 *        http://assemblysys.com/php-point-in-polygon-algorithm/
 */
int pointInPolygon(int nVert, float *vertX, float *vertY, float testX, float testY);

/** thin plate spline
 *  \param[in] affineParam        affine parameters
 *  \param[in] vectorX            vector X
 *  \param[in] localCoeff         local coefficients
 *  \param[in] numP          index k
 *  \param[in] colInd        index i
 *
 *  \retrun nothing
 *  \note https://en.wikipedia.org/wiki/Thin_plate_spline
 */
void tps(float *sigma, float *affineParam, float *vectorX, float *ctrlP, float localP, float *localCoeff, int numP, int colInd, size_t w, size_t h, size_t nc);

/** radial basis approximation
 *  \param[in] ctrlP        c_k
 *  \param[in] localP         x
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
