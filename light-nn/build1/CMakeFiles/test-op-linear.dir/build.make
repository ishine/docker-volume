# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /volume/light-nn

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /volume/light-nn/build1

# Include any dependencies generated for this target.
include CMakeFiles/test-op-linear.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/test-op-linear.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test-op-linear.dir/flags.make

CMakeFiles/test-op-linear.dir/test/test-op-linear.cc.o: CMakeFiles/test-op-linear.dir/flags.make
CMakeFiles/test-op-linear.dir/test/test-op-linear.cc.o: ../test/test-op-linear.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/volume/light-nn/build1/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/test-op-linear.dir/test/test-op-linear.cc.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test-op-linear.dir/test/test-op-linear.cc.o -c /volume/light-nn/test/test-op-linear.cc

CMakeFiles/test-op-linear.dir/test/test-op-linear.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test-op-linear.dir/test/test-op-linear.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /volume/light-nn/test/test-op-linear.cc > CMakeFiles/test-op-linear.dir/test/test-op-linear.cc.i

CMakeFiles/test-op-linear.dir/test/test-op-linear.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test-op-linear.dir/test/test-op-linear.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /volume/light-nn/test/test-op-linear.cc -o CMakeFiles/test-op-linear.dir/test/test-op-linear.cc.s

CMakeFiles/test-op-linear.dir/test/test-op-linear.cc.o.requires:

.PHONY : CMakeFiles/test-op-linear.dir/test/test-op-linear.cc.o.requires

CMakeFiles/test-op-linear.dir/test/test-op-linear.cc.o.provides: CMakeFiles/test-op-linear.dir/test/test-op-linear.cc.o.requires
	$(MAKE) -f CMakeFiles/test-op-linear.dir/build.make CMakeFiles/test-op-linear.dir/test/test-op-linear.cc.o.provides.build
.PHONY : CMakeFiles/test-op-linear.dir/test/test-op-linear.cc.o.provides

CMakeFiles/test-op-linear.dir/test/test-op-linear.cc.o.provides.build: CMakeFiles/test-op-linear.dir/test/test-op-linear.cc.o


# Object files for target test-op-linear
test__op__linear_OBJECTS = \
"CMakeFiles/test-op-linear.dir/test/test-op-linear.cc.o"

# External object files for target test-op-linear
test__op__linear_EXTERNAL_OBJECTS =

test/test-op-linear: CMakeFiles/test-op-linear.dir/test/test-op-linear.cc.o
test/test-op-linear: CMakeFiles/test-op-linear.dir/build.make
test/test-op-linear: lib/libgtest.a
test/test-op-linear: lib/liblnn.a
test/test-op-linear: lib/openblas.lib
test/test-op-linear: CMakeFiles/test-op-linear.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/volume/light-nn/build1/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable test/test-op-linear"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test-op-linear.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test-op-linear.dir/build: test/test-op-linear

.PHONY : CMakeFiles/test-op-linear.dir/build

CMakeFiles/test-op-linear.dir/requires: CMakeFiles/test-op-linear.dir/test/test-op-linear.cc.o.requires

.PHONY : CMakeFiles/test-op-linear.dir/requires

CMakeFiles/test-op-linear.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test-op-linear.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test-op-linear.dir/clean

CMakeFiles/test-op-linear.dir/depend:
	cd /volume/light-nn/build1 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /volume/light-nn /volume/light-nn /volume/light-nn/build1 /volume/light-nn/build1 /volume/light-nn/build1/CMakeFiles/test-op-linear.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/test-op-linear.dir/depend

