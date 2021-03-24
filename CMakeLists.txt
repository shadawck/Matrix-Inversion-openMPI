cmake_minimum_required (VERSION 3.6)
project (pp_tp3)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

set(CMAKE_C_COMPILER mpicc)
set(CMAKE_CXX_COMPILER mpic++)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-unused-variable -Wshadow -Wuninitialized -Wunused-label")

add_executable(pp_tp3 main.cpp Matrix.cpp)
target_link_libraries(pp_tp3 ${MPI_CXX_LIBRARIES} ${MPI_C_LIBRARIES} ${MPI_LIBRARIES})

INCLUDE(CPack)