/** \file testing.h
 *  \brief     Tests for each function
 *  \details   Unit tests
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

  #ifndef TESTING_H
  #define TESTING_H

  #include "helper.h"
  #include "shapeRegistration.h"
  #include <iostream>
  #include <stdio.h>
  #include <cstring>

  using namespace std;
  using namespace cv;

  /** Function to perform all the available tests
   *  \param[in] imgIn              array of the input image pixels
   *  \param[in] mIn                cv::Mat form of the input image
   *  \param[in] w                  width of the image
   *  \param[in] h                  height of the image
   *  \param[out] imgOut            array of the output image pixels
   *
   *  \return nothing
   */
  void test_all(float *imgIn, cv::Mat mIn, int w, int h, float *imgOut);

  #endif  // TESTING_H
