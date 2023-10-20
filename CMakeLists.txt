cmake_minimum_required(VERSION 3.15)
project(struvolpy NONE)

find_package(Python ${PYTHON_VERSION} REQUIRED COMPONENTS Interpreter)
if(WIN32)
   string(REPLACE "\\" "/" Python_SITELIB "${Python_SITELIB}")
endif(WIN32)

install(
    DIRECTORY
        struvolpy
    DESTINATION
        ${Python_SITELIB}
)
