cmake_minimum_required(VERSION 3.6)
project(HomogenizedInflatables)

# CMP0063: Honor visibility properties for all target types.
if(POLICY CMP0063)
    cmake_policy(SET CMP0063 NEW)
endif()

################################################################################

set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(CMAKE_CXX_FLAGS_RELWITHASSERT        "-O3" CACHE STRING "Flags used during RelWithAssertions builds" FORCE)
set(CMAKE_C_FLAGS_RELWITHASSERT          "-O3" CACHE STRING "Flags used during RelWithAssertions builds" FORCE)
set(CMAKE_EXE_LINKER_FLAGS_RELWITHASSERT    "" CACHE STRING "Flags used during RelWithAssertions builds" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_RELWITHASSERT "" CACHE STRING "Flags used during RelWithAssertions builds" FORCE)
set(CMAKE_MODULE_LINKER_FLAGS_RELWITHASSERT "" CACHE STRING "Flags used during RelWithAssertions builds" FORCE)
set(CMAKE_STATIC_LINKER_FLAGS_RELWITHASSERT "" CACHE STRING "Flags used during RelWithAssertions builds" FORCE)
mark_as_advanced(
    CMAKE_CXX_FLAGS_RELWITHASSERT
    CMAKE_C_FLAGS_RELWITHASSERT
    CMAKE_EXE_LINKER_FLAGS_RELWITHASSERT
    CMAKE_SHARED_LINKER_FLAGS_RELWITHASSERT
    CMAKE_MODULE_LINKER_FLAGS_RELWITHASSERT
    CMAKE_STATIC_LINKER_FLAGS_RELWITHASSERT
    )

set(CMAKE_CXX_FLAGS_RELEASENATIVE        "-O3 -march=native -DNDEBUG" CACHE STRING "Flags used during native release builds" FORCE)
set(CMAKE_C_FLAGS_RELEASENATIVE          "-O3 -march=native -DNDEBUG" CACHE STRING "Flags used during native release builds" FORCE)
set(CMAKE_EXE_LINKER_FLAGS_RELEASENATIVE    "" CACHE STRING "Flags used during native release builds" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_RELEASENATIVE "" CACHE STRING "Flags used during native release builds" FORCE)
set(CMAKE_MODULE_LINKER_FLAGS_RELEASENATIVE "" CACHE STRING "Flags used during native release builds" FORCE)
set(CMAKE_STATIC_LINKER_FLAGS_RELEASENATIVE "" CACHE STRING "Flags used during native release builds" FORCE)
mark_as_advanced(
    CMAKE_CXX_FLAGS_RELEASENATIVE
    CMAKE_C_FLAGS_RELEASENATIVE
    CMAKE_EXE_LINKER_FLAGS_RELEASENATIVE
    CMAKE_SHARED_LINKER_FLAGS_RELEASENATIVE
    CMAKE_MODULE_LINKER_FLAGS_RELEASENATIVE
    CMAKE_STATIC_LINKER_FLAGS_RELEASENATIVE
    )

set(CMAKE_CXX_FLAGS_DEBUGNATIVE        "-g -march=native" CACHE STRING "Flags used during native debug builds" FORCE)
set(CMAKE_C_FLAGS_DEBUGNATIVE          "-g -march=native" CACHE STRING "Flags used during native debug builds" FORCE)
set(CMAKE_EXE_LINKER_FLAGS_DEBUGNATIVE    "" CACHE STRING "Flags used during native debug builds" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_DEBUGNATIVE "" CACHE STRING "Flags used during native debug builds" FORCE)
set(CMAKE_MODULE_LINKER_FLAGS_DEBUGNATIVE "" CACHE STRING "Flags used during native debug builds" FORCE)
set(CMAKE_STATIC_LINKER_FLAGS_DEBUGNATIVE "" CACHE STRING "Flags used during native debug builds" FORCE)
mark_as_advanced(
    CMAKE_CXX_FLAGS_DEBUGNATIVE
    CMAKE_C_FLAGS_DEBUGNATIVE
    CMAKE_EXE_LINKER_FLAGS_DEBUGNATIVE
    CMAKE_SHARED_LINKER_FLAGS_DEBUGNATIVE
    CMAKE_MODULE_LINKER_FLAGS_DEBUGNATIVE
    CMAKE_STATIC_LINKER_FLAGS_DEBUGNATIVE
    )

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, default to RelWithAssert")
    set(CMAKE_BUILD_TYPE "RelWithAssert")
endif()

set(CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}" CACHE STRING "Choose the type of build. Options are: None Debug Release RelWithDebInfo MinSizeRel RelWithAssert" FORCE)

# Color output
include(UseColors)

if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    # Make sure warnings/errors are still colorized when using Ninja for building.
    add_definitions(-fdiagnostics-color=always)
    # Output full backtrace for compiler errors.
    # add_definitions(-ftemplate-backtrace-limit=0)
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    # -march=native is leading to an internal compiler error on GCC 9.2.1, the version shipped with Ubuntu 19.10
else()
    # # Equivalent to -march=native for Intel
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -xHost")

    # EPFL's fidis cluster only supports AVX2 even though
    # helvetios has AVX512; make a build that works on all the clusters.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -xAVX2")
endif()

# Export compile flags(used for autocompletion of the C++ code)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

# CMake plugin for vscode
include(CMakeToolsHelpers OPTIONAL)

# Enable more warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic -Wno-comment")

# We need -fPIC when compiling our libraries and our dependencies for
# the python bindings to link.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(INFLATABLES_LONG_DOUBLE "Use 'long double' as the real number type for inflation" OFF)
# find_package(MPFR REQUIRED)

# Explicitly set LIBIGL directory before adding Inflatables
set(LIBIGL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/Inflatables/3rdparty/libigl)
list(APPEND CMAKE_PREFIX_PATH ${LIBIGL_DIR})

add_subdirectory(${THIRD_PARTY_DIR}/Inflatables)
# Ensure the new repo can find the headers from Inflatables
include_directories(${THIRD_PARTY_DIR}/Inflatables)
################################################################################
# Inflation Periodic Patch Library
################################################################################
add_library(homogenized_inflatable_lib InflatablePeriodicUnit.cc InflatableMidSurfacePeriodicUnit.cc periodic_inflation_newton.cc periodic_stiffness_analysis.cc)
set_target_properties(homogenized_inflatable_lib PROPERTIES CXX_STANDARD 14)
set_target_properties(homogenized_inflatable_lib PROPERTIES CXX_STANDARD_REQUIRED ON)
target_include_directories(homogenized_inflatable_lib PUBLIC .)
# target_link_libraries(homogenized_inflatable_lib MeshFEM geometry_lib ${MPFR_LIBRARIES})
target_link_libraries(homogenized_inflatable_lib MeshFEM inflation_lib warnings::all)

################################################################################


if(INFLATABLES_LONG_DOUBLE)
    target_compile_definitions(homogenized_inflatable_lib PUBLIC -DMESHFEM_BIND_LONG_DOUBLE)
endif()

################################################################################
# Binaries
################################################################################
add_subdirectory(python_bindings)

if(NOT TARGET Catch2::Catch2)
    meshfem_download_catch()
    add_subdirectory(${MESHFEM_EXTERNAL}/Catch2)
    list(APPEND CMAKE_MODULE_PATH ${MESHFEM_EXTERNAL}/Catch2/contrib)
endif()
