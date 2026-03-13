# - Find Knitro
#  Searches for includes/libraries using environment variable $KNITRO_PATH or $KNITRO_DIR
#  KNITRO_INCLUDE_DIRS - where to find knitro.h and (separately) the c++ interface
#  KNITRO_LIBRARIES    - List of libraries needed to use knitro.
#  KNITRO_FOUND        - True if knitro found.


IF (KNITRO_INCLUDE_DIRS)
  # Already in cache, be silent
  SET (knitro_FIND_QUIETLY TRUE)
ENDIF (KNITRO_INCLUDE_DIRS)

FIND_PATH(KNITRO_INCLUDE_DIR knitro.h
	HINTS
        $ENV{KNITRO_PATH}/include
        $ENV{KNITRO_DIR}/include
        $ENV{KNITRODIR}/include
)

FIND_LIBRARY (KNITRO_LIBRARY NAMES knitro knitro1300 knitro1240 knitro1031
	HINTS
        $ENV{KNITRO_PATH}/lib
        $ENV{KNITRO_DIR}/lib
        $ENV{KNITRODIR}/lib
)

# handle the QUIETLY and REQUIRED arguments and set KNITRO_FOUND to TRUE if
# all listed variables are TRUE
INCLUDE (FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS (KNITRO DEFAULT_MSG
  KNITRO_LIBRARY
  KNITRO_INCLUDE_DIR)


IF(KNITRO_FOUND)
    SET (KNITRO_LIBRARIES ${KNITRO_LIBRARY})
    SET (KNITRO_INCLUDE_DIRS "${KNITRO_INCLUDE_DIR}" "${KNITRO_INCLUDE_DIR}/../examples/C++/include")
    SET (KNITRO_CPP_Examples "${KNITRO_INCLUDE_DIR}/../examples/C++")

    MESSAGE("   Knitro lib path    : ${KNITRO_LIBRARIES}")
    MESSAGE("   Knitro include path: ${KNITRO_INCLUDE_DIRS}")
    IF (EXISTS "${KNITRO_CPP_Examples}/src")
        file(GLOB INTERFACE_SOURCES ${KNITRO_CPP_Examples}/src/*.cpp)
        add_library(knitrocpp SHARED ${INTERFACE_SOURCES})
        target_link_libraries(knitrocpp PUBLIC ${KNITRO_LIBRARIES})
        target_include_directories(knitrocpp PUBLIC SYSTEM ${KNITRO_INCLUDE_DIRS})
        set_property(TARGET knitrocpp PROPERTY CXX_STANDARD 11)
        add_library(knitro::knitro ALIAS knitrocpp)
        target_compile_definitions(knitrocpp PUBLIC -DHAS_KNITRO)
        MESSAGE("Using newer knitro!")
    ELSE()
        add_library(knitrocpp INTERFACE ${INTERFACE_SOURCES})
        target_link_libraries(knitrocpp INTERFACE ${KNITRO_LIBRARIES})
        target_include_directories(knitrocpp INTERFACE SYSTEM ${KNITRO_INCLUDE_DIRS})
        target_compile_definitions(knitrocpp INTERFACE -DHAS_KNITRO -DKNITRO_LEGACY)
        add_library(knitro::knitro ALIAS knitrocpp)
        MESSAGE("Using legacy knitro!")
    ENDIF()
ELSE (KNITRO_FOUND)
    SET (KNITRO_LIBRARIES)
ENDIF (KNITRO_FOUND)

MARK_AS_ADVANCED (KNITRO_LIBRARY KNITRO_INCLUDE_DIR KNITRO_INCLUDE_DIRS KNITRO_LIBRARIES)
