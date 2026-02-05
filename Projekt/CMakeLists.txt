cmake_minimum_required(VERSION 3.10) # setting this is required
project(CMAKE_CURRENT_LIST_DIR)                 # this sets the project name

file(GLOB_RECURSE sources      src/*.c src/*.h)
file(GLOB_RECURSE sources_test test/*.c)

add_executable(CMAKE_CURRENT_LIST_DIR ${sources} ${data})

target_compile_options(CMAKE_CURRENT_LIST_DIR PUBLIC -Wall -Wfloat-conversion)

target_include_directories(CMAKE_CURRENT_LIST_DIR PUBLIC src )

target_link_libraries(CMAKE_CURRENT_LIST_DIR json-c curl)

file(COPY ${data} DESTINATION resources)

find_package(GTest)

if(GTEST_FOUND)
  add_executable(unit_tests ${sources_test} ${sources})

  target_compile_definitions(unit_tests PUBLIC UNIT_TESTS)

  set_target_properties(CMAKE_CURRENT_LIST_DIR PROPERTIES ENABLE_EXPORTS on)

  target_link_libraries(unit_tests PUBLIC
    ${GTEST_BOTH_LIBRARIES}
    CMAKE_CURRENT_LIST_DIR
  )

endif()

install(TARGETS CMAKE_CURRENT_LIST_DIR DESTINATION example_destination)
install(DIRECTORY resources DESTINATION example_destination)

set(CPACK_PACKAGE_NAME CMAKE_CURRENT_LIST_DIR)
set(CPACK_PACKAGE_VERSION "1.0.0")

set(CPACK_MONOLITHIC_INSTALL 1)

include(CPack)
