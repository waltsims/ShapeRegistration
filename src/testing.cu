/**
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

  #include "testing.h"
  #define DIM_C_REF 5

  void test_all(float *imgIn, cv::Mat mIn, int w, int h, float *imgOut){
    printf("Performing all the available tests...\n");

    // Test: cutMargins() [OK]
    // Note: the input image circle.png is shifted up.
    printf("---Testing the cutMargins()-----------------------\n");
    float *resizedImg;
    float *resizedImOut;
    int resizedW;
    int resizedH;
    Margins margins;

    cutMargins(imgIn, w, h, resizedImg, resizedW, resizedH, margins);

    printf("Dimensions of the original image: width: %d, height: %d.\n", w, h);
    printf(
        "Margins positions (starting from 0): top: %d, bottom: %d, left: %d, "
        "right: %d.\n",
        margins.top, margins.bottom, margins.left, margins.right);
    printf("Dimensions of the cropped image: width: %d, height: %d.\n", resizedW,
           resizedH);
    printf("Background margin sizes: top: %d, bottom: %d, left: %d, right: %d.\n",
           margins.top, h - (margins.bottom + 1), margins.left,
           w - (margins.right + 1));
    printf("--------------------------------------------------\n\n");

    cv::Mat resizedImgOut(resizedH, resizedW, CV_32FC1);
    convert_layered_to_mat(resizedImgOut, resizedImg);
    showImage("Cropped Output", resizedImgOut, 100 + w, 100);
    // end of test: cutMargins().

    // Test: addMargins() [OK]
    printf("---Testing the addMargins()-----------------------\n");
    float *recoveredImg = new float[(size_t)w * h];
    addMargins(resizedImg, resizedW, resizedH, recoveredImg, w, h, margins);
    printf(
        "Showing the difference of the original and the recovered image (should "
        "be background)...\n");
    cv::Mat mRecov(h, w, CV_32FC1);
    convert_layered_to_mat(mRecov, recoveredImg);
    showImage("Recovered vs original image", cv::abs(mIn - mRecov), 300, 300);
    printf("--------------------------------------------------\n\n");
    // end of test: addMargins().

    // Test: centerOfMass() [OK]
    printf("---Testing the centerOfMass()---------------------\n");
    float xCentCoord;  // x-coordinate of the center of mass
    float yCentCoord;  // y-coordinate of the center of mass
    centerOfMass(imgIn, w, h, xCentCoord, yCentCoord);
    printf("Center of mass of the original image (starting from 0): ( %f, %f)\n",
           xCentCoord, yCentCoord);
    centerOfMass(resizedImg, resizedW, resizedH, xCentCoord, yCentCoord);
    printf("Center of mass of the cropped image (starting from 0): ( %f, %f)\n",
           xCentCoord, yCentCoord);
    printf("--------------------------------------------------\n\n");
    // end of test: centerOfMass().

    // Test: setPixelCoords() before normalization [OK]
    printf("---Testing the setPixelCoords() [before Norm]-----\n");
    PixelCoords *pCoords = new PixelCoords[resizedW * resizedH];
    setPixelCoords(pCoords, resizedW, resizedH);
    printf("pCoords[0].x = %f\n", pCoords[0].x);
    printf("pCoords[0].y = %f\n", pCoords[0].y);
    int lastIndex = (resizedW - 1) + (resizedH - 1) * resizedW;
    printf("pCoords[last].x = %f\n", pCoords[lastIndex].x);
    printf("pCoords[last].y = %f\n", pCoords[lastIndex].y);
    float minPCoordX = pCoords[0].x, minPCoordY = pCoords[0].y;
    float maxPCoordX = pCoords[0].x, maxPCoordY = pCoords[0].y;
    int minPCoordXInd = 0, minPCoordYInd = 0;
    int maxPCoordXInd = 0, maxPCoordYInd = 0;
    int index;
    /** Find the minimum and maximum pixel coordinates per x,y
     *  and the respective x,y indices */
    for (int y = 0; y < resizedH; y++) {
      for (int x = 0; x < resizedW; x++) {
        index = x + y * resizedW;
        if (pCoords[index].x < minPCoordX) {
          minPCoordX = pCoords[index].x;
          minPCoordXInd = x;
        }
        if (pCoords[index].x > maxPCoordX) {
          maxPCoordX = pCoords[index].x;
          maxPCoordXInd = x;
        }
        if (pCoords[index].y < minPCoordY) {
          minPCoordY = pCoords[index].x;
          minPCoordYInd = y;
        }
        if (pCoords[index].y > maxPCoordY) {
          maxPCoordY = pCoords[index].y;
          maxPCoordYInd = y;
        }
      }
    }
    printf(
        "Minimum/Maximum pixel coordinates:\nminX: %4.f (for x: %d), maxX: %4.f "
        "(for x: %d)\nminY: %4.f (for y: %d), maxY: %4.f (for y: %d) \n",
        minPCoordX, minPCoordXInd, maxPCoordX, maxPCoordXInd, minPCoordY,
        minPCoordYInd, maxPCoordY, maxPCoordYInd);
    printf("--------------------------------------------------\n\n");
    // end of test: setPixelCoords() before normalization.

    // Test: setQuadCoords() before normalization [OK]
    printf("---Testing the setQuadCoords() [before Norm]------\n");
    QuadCoords *qCoords = new QuadCoords[resizedW * resizedH];
    setQuadCoords(qCoords, resizedW, resizedH);
    printf(
        "qCoords[0].x[0] = %f, qCoords[0].x[1] = %f, qCoords[0].x[2] = %f, "
        "qCoords[0].x[3] = %f\n",
        qCoords[0].x[0], qCoords[0].x[1], qCoords[0].x[2], qCoords[0].x[3]);
    printf(
        "qCoords[0].y[0] = %f, qCoords[0].y[1] = %f, qCoords[0].y[2] = %f, "
        "qCoords[0].y[3] = %f\n",
        qCoords[0].y[0], qCoords[0].y[1], qCoords[0].y[2], qCoords[0].y[3]);
    printf(
        "qCoords[last].x[0] = %f, qCoords[last].x[1] = %f, qCoords[last].x[2] = "
        "%f, qCoords[last].x[3] = %f\n",
        qCoords[lastIndex].x[0], qCoords[lastIndex].x[1], qCoords[lastIndex].x[2],
        qCoords[lastIndex].x[3]);
    printf(
        "qCoords[last].y[0] = %f, qCoords[last].y[1] = %f, qCoords[last].y[2] = "
        "%f, qCoords[last].y[3] = %f\n",
        qCoords[lastIndex].y[0], qCoords[lastIndex].y[1], qCoords[lastIndex].y[2],
        qCoords[lastIndex].y[3]);
    printf("--------------------------------------------------\n\n");
    // end of test: setQuadCoords() before normalization.

    // Test: pCoordsNormalization() [OK]
    printf("---Testing the pCoordsNormalization()-------------\n");
    pCoordsNormalization(resizedW, resizedH, pCoords, xCentCoord, yCentCoord);
    printf("pCoords[0].x = %f\n", pCoords[0].x);
    printf("pCoords[0].y = %f\n", pCoords[0].y);
    printf("pCoords[last].x = %f\n", pCoords[lastIndex].x);
    printf("pCoords[last].y = %f\n", pCoords[lastIndex].y);
    minPCoordX = pCoords[0].x, minPCoordY = pCoords[0].y;
    maxPCoordX = pCoords[0].x, maxPCoordY = pCoords[0].y;
    minPCoordXInd = 0, minPCoordYInd = 0;
    maxPCoordXInd = 0, maxPCoordYInd = 0;
    /** Find the minimum and maximum pixel coordinates per x,y
     *  and the respective x,y indices */
    for (int y = 0; y < resizedH; y++) {
      for (int x = 0; x < resizedW; x++) {
        index = x + y * resizedW;
        if (pCoords[index].x < minPCoordX) {
          minPCoordX = pCoords[index].x;
          minPCoordXInd = x;
        }
        if (pCoords[index].x > maxPCoordX) {
          maxPCoordX = pCoords[index].x;
          maxPCoordXInd = x;
        }
        if (pCoords[index].y < minPCoordY) {
          minPCoordY = pCoords[index].x;
          minPCoordYInd = y;
        }
        if (pCoords[index].y > maxPCoordY) {
          maxPCoordY = pCoords[index].y;
          maxPCoordYInd = y;
        }
      }
    }
    printf(
        "Minimum/Maximum pixel coordinates:\nminX: %f (for x: %d), maxX: %f (for "
        "x: %d)\nminY: %f (for y: %d), maxY: %f (for y: %d) \n",
        minPCoordX, minPCoordXInd, maxPCoordX, maxPCoordXInd, minPCoordY,
        minPCoordYInd, maxPCoordY, maxPCoordYInd);
    printf("--------------------------------------------------\n\n");
    // end of test: pCoordsNormalization().

    // Test: qCoordsNormalization() [OK]
    printf("---Testing the qCoordsNormalization()-------------\n");
    qCoordsNormalization(resizedW, resizedH, qCoords, xCentCoord, yCentCoord);
    printf(
        "qCoords[0].x[0] = %f, qCoords[0].x[1] = %f, qCoords[0].x[2] = %f, "
        "qCoords[0].x[3] = %f\n",
        qCoords[0].x[0], qCoords[0].x[1], qCoords[0].x[2], qCoords[0].x[3]);
    printf(
        "qCoords[0].y[0] = %f, qCoords[0].y[1] = %f, qCoords[0].y[2] = %f, "
        "qCoords[0].y[3] = %f\n",
        qCoords[0].y[0], qCoords[0].y[1], qCoords[0].y[2], qCoords[0].y[3]);
    printf(
        "qCoords[last].x[0] = %f, qCoords[last].x[1] = %f, qCoords[last].x[2] = "
        "%f, qCoords[last].x[3] = %f\n",
        qCoords[lastIndex].x[0], qCoords[lastIndex].x[1], qCoords[lastIndex].x[2],
        qCoords[lastIndex].x[3]);
    printf(
        "qCoords[last].y[0] = %f, qCoords[last].y[1] = %f, qCoords[last].y[2] = "
        "%f, qCoords[last].y[3] = %f\n",
        qCoords[lastIndex].y[0], qCoords[lastIndex].y[1], qCoords[lastIndex].y[2],
        qCoords[lastIndex].y[3]);
    printf("--------------------------------------------------\n\n");
    // end of test: qCoordsNormalization().

    // Test: pCoordsDenormalization() [OK]
    printf("---Testing the pCoordsDenormalization()-------------\n");
    pCoordsDenormalization(resizedW, resizedH, pCoords, xCentCoord, yCentCoord);
    printf("pCoords[0].x = %f\n", pCoords[0].x);
    printf("pCoords[0].y = %f\n", pCoords[0].y);
    printf("pCoords[last].x = %f\n", pCoords[lastIndex].x);
    printf("pCoords[last].y = %f\n", pCoords[lastIndex].y);
    minPCoordX = pCoords[0].x, minPCoordY = pCoords[0].y;
    maxPCoordX = pCoords[0].x, maxPCoordY = pCoords[0].y;
    minPCoordXInd = 0, minPCoordYInd = 0;
    maxPCoordXInd = 0, maxPCoordYInd = 0;
    /** Find the minimum and maximum pixel coordinates per x,y
     *  and the respective x,y indices */
    for (int y = 0; y < resizedH; y++) {
      for (int x = 0; x < resizedW; x++) {
        index = x + y * resizedW;
        if (pCoords[index].x < minPCoordX) {
          minPCoordX = pCoords[index].x;
          minPCoordXInd = x;
        }
        if (pCoords[index].x > maxPCoordX) {
          maxPCoordX = pCoords[index].x;
          maxPCoordXInd = x;
        }
        if (pCoords[index].y < minPCoordY) {
          minPCoordY = pCoords[index].x;
          minPCoordYInd = y;
        }
        if (pCoords[index].y > maxPCoordY) {
          maxPCoordY = pCoords[index].y;
          maxPCoordYInd = y;
        }
      }
    }
    printf(
        "Minimum/Maximum pixel coordinates:\nminX: %f (for x: %d), maxX: %f (for "
        "x: %d)\nminY: %f (for y: %d), maxY: %f (for y: %d) \n",
        minPCoordX, minPCoordXInd, maxPCoordX, maxPCoordXInd, minPCoordY,
        minPCoordYInd, maxPCoordY, maxPCoordYInd);
    printf("--------------------------------------------------\n\n");
    // end of test: pCoordsDenormalization().

    TPSParams tpsParams;

    // Test: pTPS() [in progress]
    printf("---Testing the pTPS()---------------------------\n");
    pTPS(resizedW, resizedH, pCoords, tpsParams, DIM_C_REF);
    printf("pCoords[0].x = %f\n", pCoords[0].x);
    printf("pCoords[0].y = %f\n", pCoords[0].y);
    printf("pCoords[last].x = %f\n", pCoords[lastIndex].x);
    printf("pCoords[last].y = %f\n", pCoords[lastIndex].y);
    minPCoordX = pCoords[0].x, minPCoordY = pCoords[0].y;
    maxPCoordX = pCoords[0].x, maxPCoordY = pCoords[0].y;
    minPCoordXInd = 0, minPCoordYInd = 0;
    maxPCoordXInd = 0, maxPCoordYInd = 0;
    /** Find the minimum and maximum pixel coordinates per x,y
     *  and the respective x,y indices */
    for (int y = 0; y < resizedH; y++) {
      for (int x = 0; x < resizedW; x++) {
        index = x + y * resizedW;
        if (pCoords[index].x < minPCoordX) {
          minPCoordX = pCoords[index].x;
          minPCoordXInd = x;
        }
        if (pCoords[index].x > maxPCoordX) {
          maxPCoordX = pCoords[index].x;
          maxPCoordXInd = x;
        }
        if (pCoords[index].y < minPCoordY) {
          minPCoordY = pCoords[index].x;
          minPCoordYInd = y;
        }
        if (pCoords[index].y > maxPCoordY) {
          maxPCoordY = pCoords[index].y;
          maxPCoordYInd = y;
        }
      }
    }
    printf(
        "Minimum/Maximum pixel coordinates:\nminX: %f (for x: %d), maxX: %f (for "
        "x: %d)\nminY: %f (for y: %d), maxY: %f (for y: %d) \n",
        minPCoordX, minPCoordXInd, maxPCoordX, maxPCoordXInd, minPCoordY,
        minPCoordYInd, maxPCoordY, maxPCoordYInd);
    printf("--------------------------------------------------\n\n");
    // end of test: pTPS().


    // Test: qTPS() [in progress]
    printf("---Testing the qTPS()---------------------------\n");
    qTPS(resizedW, resizedH, qCoords, tpsParams, DIM_C_REF);

    printf(
        "qCoords[0].x[0] = %f, qCoords[0].x[1] = %f, qCoords[0].x[2] = %f, "
        "qCoords[0].x[3] = %f\n",
        qCoords[0].x[0], qCoords[0].x[1], qCoords[0].x[2], qCoords[0].x[3]);
    printf(
        "qCoords[0].y[0] = %f, qCoords[0].y[1] = %f, qCoords[0].y[2] = %f, "
        "qCoords[0].y[3] = %f\n",
        qCoords[0].y[0], qCoords[0].y[1], qCoords[0].y[2], qCoords[0].y[3]);
    printf(
        "qCoords[last].x[0] = %f, qCoords[last].x[1] = %f, qCoords[last].x[2] = "
        "%f, qCoords[last].x[3] = %f\n",
        qCoords[lastIndex].x[0], qCoords[lastIndex].x[1], qCoords[lastIndex].x[2],
        qCoords[lastIndex].x[3]);
    printf(
        "qCoords[last].y[0] = %f, qCoords[last].y[1] = %f, qCoords[last].y[2] = "
        "%f, qCoords[last].y[3] = %f\n",
        qCoords[lastIndex].y[0], qCoords[lastIndex].y[1], qCoords[lastIndex].y[2],
        qCoords[lastIndex].y[3]);
    printf("--------------------------------------------------\n\n");
    // end of test: qTPS().

    resizedImOut = new float[resizedW * resizedH];
    // transpose(resizedImg, pCoords, qCoords, resizedW, resizedH, resizedImOut);

    convert_layered_to_mat(resizedImgOut, resizedImOut);
    showImage("Resized Output", resizedImgOut, 100 + w + 40 + w + 40, 100);

    delete[] resizedImg;
    delete[] recoveredImg;
  }
