# Nonlinear Shape Registration without Correspondences
## a project for TUM course GPU Programming in Computer vision

## Authors (in alphabetic order)
- Gerasimos Chourdakis
- Sungjae Jung
- Walter Simson

## Libraries

The following libraries are required

(coming soon)

## Setup

For development you should install the following

* `clang` : We use `clang's `clang-format` to keep our coding style consistent
* `cmake` : We use `cmake` to generate user friendly make file and help with 
on multiple operating systems

```sh

#To create a Makefile with CMake run the folling in the base project directory

cmake . 

#This process need only be repeated if CMakeList.txt has been modified

#Once the Make file is generated, you can make the project.

make
```

To compile the 

##Style Guide

We use [google's C++ style guide](http://google.github.io/styleguide/cppguide.html). It
is a reasonable guideline, used by a large organization to great success and at
the same time it is an interesting read and can maybe even teach you something
about C++.

You can use the script in `scripts/format` to automatically reformat all source
files. Alternatively try one of the integrations of `clang-format` into various
editors/IDEs. ( comming soon )

Much of this document was based on the knowledge passed on by @cqql
