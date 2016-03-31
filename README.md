# Nonlinear Shape Registration without Correspondences
Student project for the course *GPU Programming in Computer Vision* (Technical University of Munich, WS2015-16)

## Authors (in alphabetic order)
- Gerasimos Chourdakis
- Sungjae Jung
- Walter Simson

## Requirements

The following libraries are required

* CUDA compiler - for GPU-enabled computations.
* OpenCV - for image I/O.
* lmfit (included in the 3rdparty/ directory) - for solving the nonlinear system of equations using the Levenberg-Marquardt method.

Instead of OpenCV, support for the libpng library may be added later.

## Setup

We use `cmake` to generate a user friendly Makefile and to support various operating system setups.

To create a Makefile with CMake run the folling in the base project directory:
```sh
    cmake . 
```
Please mind the period at the end. This process needs to be repeated only if the CMakeList.txt has been modified.

Once the Makefile is generated, you can build the project:
```sh
    make
```


##Style Guide

For development you should install `clang`. We use `clang`'s `clang-format` to keep our coding style consistent.

We also use [google's C++ style guide](http://google.github.io/styleguide/cppguide.html). It
is a reasonable guideline, used by a large organization to great success and at
the same time it is an interesting read and can maybe even teach you something
about C++.

You can use the script `scripts/format` to automatically reformat all the source
files. Alternatively, try one of the integrations of `clang-format` into various
editors/IDEs. ( comming soon )


##Aknowledgements
Our code was created using the framework provided in the course as a starting point.

Much of this document was based on the knowledge passed on by @cqql.

