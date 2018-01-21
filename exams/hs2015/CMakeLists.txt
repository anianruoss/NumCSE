project(ProblemSheet7)
cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 14)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING
        "Build type, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE)

# Use clang or gcc, 
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    # Using Clang
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wno-deprecated-declarations" )
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -Wall -Wextra -pedantic")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -fast-math")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    # Using GCC
    # Use -O3 for strong optimization. We use stdc++11. Compile with Wall to get all warnings.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wno-deprecated-declarations" )
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -Wall -Wextra -pedantic")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -fast-math")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules)

# We find Eigen3
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

### MAKE RULES FOR INDIVIDUAL EXERCISES
# Make rule for problem1 (type make problem1)
set(SRCS problem1.cpp)
add_executable(problem1 ${SRCS})

# Make rule for problem2 (type make problem2)
set(SRCS problem2.cpp)
add_executable(problem2 ${SRCS})

# Make rule for problem3 (type make problem3)
set(SRCS problem3.cpp)
add_executable(problem3 ${SRCS})

# Make rule for problem5 (type make problem5)
set(SRCS problem5.cpp)
add_executable(problem5 ${SRCS})

