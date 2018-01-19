project(ExamJanuary17)
cmake_minimum_required(VERSION 2.6.2)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING
        "Build type, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE)

# Use clang or gcc,
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    # Using Clang
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wno-deprecated-declarations -Wno-unused-variable" )
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -Wall -Wextra -pedantic")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g -O3 -fast-math -Wall -Wextra -pedantic")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    # Using GCC
    # Use -O3 for strong optimization. We use stdc++11. Compile with Wall to get all warnings.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wno-deprecated-declarations -Wno-unused-variable" )
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -Wall -Wextra -pedantic")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g -O3 -fast-math -Wall -Wextra -pedantic")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules)

# Find Eigen3
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

macro(add_problem name)
    set(SRCS ${name}.cpp)
    add_executable(template_${name} ${SRCS})
    target_compile_definitions(template_${name} PRIVATE SOLUTION=0 INTERNAL=1)
    add_executable(solution_${name} ${SRCS})
    target_compile_definitions(solution_${name} PRIVATE SOLUTION=1 INTERNAL=1)
endmacro(add_problem)

### MAKE RULES FOR INDIVIDUAL EXERCISES
add_problem(problemX_NewtonVector)
add_problem(problemX_rankk)
add_problem(problemX_singlestep)
add_problem(problemX_polar)

