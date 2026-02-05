#!/bin/bash

# Sets up the C/C++ project using the necessary structures of the pitchfork layout (PFL), using CMake as build system.
# Assumes small to medium sized projects, so we omit the optional submodule and extra script support.
# Sets up the optional tests, and docs directories.
# https://joholl.github.io/pitchfork-website/


mkdir build
mkdir src 
mkdir include
mkdir tests 
mkdir docs 


