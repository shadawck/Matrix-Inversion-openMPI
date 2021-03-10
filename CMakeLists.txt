cmake_minimum_required (VERSION 2.6)
project (pp_tp3)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -O3 -fopenmp -Wno-unused-variable -Wshadow -Wuninitialized -Wunused-label ${OpenMP_CXX_FLAGS}")

find_package(MPI REQUIRED)
add_definitions(-DOMPI_SKIP_MPICXX)

add_executable(main.cpp Matrix.cpp MpiUtils.cpp MpiUtils.h)
target_link_libraries(pp_tp3 ${MPI_C_LIBRARIES})

INCLUDE(CPack)
