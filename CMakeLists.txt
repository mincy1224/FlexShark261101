cmake_minimum_required(VERSION 3.16)
project(shark)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-write-strings -Wno-unused-result -maes -Wno-ignored-attributes -march=native -Wno-deprecated-declarations")

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package (Eigen3 REQUIRED NO_MODULE)

add_subdirectory(ext/cryptoTools)
add_subdirectory(ext/cnpy)

add_library(${PROJECT_NAME}
    src/utils/timer.cpp
    src/utils/comm.cpp
    src/utils/lookuptable.cpp
    src/protocols/common.cpp
    src/protocols/init.cpp
    src/protocols/finalize.cpp
    src/protocols/input.cpp
    src/protocols/output.cpp
    src/protocols/matmul.cpp
    src/protocols/drelu.cpp
    src/protocols/b2a.cpp
    src/protocols/select.cpp
    src/protocols/relu.cpp
    src/protocols/conv.cpp
    src/protocols/maxpool.cpp
    src/protocols/sumpool.cpp
    src/protocols/mul.cpp
    src/protocols/lut.cpp
    src/protocols/truncate.cpp
    src/protocols/down.cpp
    src/protocols/up.cpp
    src/protocols/matmulpar.cpp
    src/crypto/dcfbit.cpp
    src/crypto/dcfring.cpp
    src/crypto/dpfring.cpp
)

target_link_libraries (${PROJECT_NAME} Eigen3::Eigen cryptoTools)
target_include_directories(${PROJECT_NAME}
PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)


# Application entry point, isolated under app261101/

add_executable(run_lenet5
    app261101/src/run_lenet5.cpp
    app261101/src/fs_lenet5.cpp
    app261101/src/fs_scaled_tanh.cpp
)
target_include_directories(run_lenet5
    PRIVATE ${PROJECT_SOURCE_DIR}/app261101/include
)

target_link_libraries(run_lenet5 PRIVATE ${PROJECT_NAME} cnpy)

# add_executable(run_bert
#     app261101/src/run_bert.cpp
#     app261101/src/fs_bert.cpp
# )
# target_include_directories(run_bert
#     PRIVATE ${PROJECT_SOURCE_DIR}/app261101/include
# )
# target_link_libraries(run_bert ${PROJECT_NAME})

