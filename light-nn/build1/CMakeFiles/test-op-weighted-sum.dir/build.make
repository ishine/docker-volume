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
include CMakeFiles/test-op-weighted-sum.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/test-op-weighted-sum.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test-op-weighted-sum.dir/flags.make

CMakeFiles/test-op-weighted-sum.dir/test/test-op-weighted-sum.cc.o: CMakeFiles/test-op-weighted-sum.dir/flags.make
CMakeFiles/test-op-weighted-sum.dir/test/test-op-weighted-sum.cc.o: ../test/test-op-weighted-sum.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/volume/light-nn/build1/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/test-op-weighted-sum.dir/test/test-op-weighted-sum.cc.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test-op-weighted-sum.dir/test/test-op-weighted-sum.cc.o -c /volume/light-nn/test/test-op-weighted-sum.cc

CMakeFiles/test-op-weighted-sum.dir/test/test-op-weighted-sum.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test-op-weighted-sum.dir/test/test-op-weighted-sum.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /volume/light-nn/test/test-op-weighted-sum.cc > CMakeFiles/test-op-weighted-sum.dir/test/test-op-weighted-sum.cc.i

CMakeFiles/test-op-weighted-sum.dir/test/test-op-weighted-sum.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test-op-weighted-sum.dir/test/test-op-weighted-sum.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /volume/light-nn/test/test-op-weighted-sum.cc -o CMakeFiles/test-op-weighted-sum.dir/test/test-op-weighted-sum.cc.s

CMakeFiles/test-op-weighted-sum.dir/test/test-op-weighted-sum.cc.o.requires:

.PHONY : CMakeFiles/test-op-weighted-sum.dir/test/test-op-weighted-sum.cc.o.requires

CMakeFiles/test-op-weighted-sum.dir/test/test-op-weighted-sum.cc.o.provides: CMakeFiles/test-op-weighted-sum.dir/test/test-op-weighted-sum.cc.o.requires
	$(MAKE) -f CMakeFiles/test-op-weighted-sum.dir/build.make CMakeFiles/test-op-weighted-sum.dir/test/test-op-weighted-sum.cc.o.provides.build
.PHONY : CMakeFiles/test-op-weighted-sum.dir/test/test-op-weighted-sum.cc.o.provides

CMakeFiles/test-op-weighted-sum.dir/test/test-op-weighted-sum.cc.o.provides.build: CMakeFiles/test-op-weighted-sum.dir/test/test-op-weighted-sum.cc.o


# Object files for target test-op-weighted-sum
test__op__weighted__sum_OBJECTS = \
"CMakeFiles/test-op-weighted-sum.dir/test/test-op-weighted-sum.cc.o"

# External object files for target test-op-weighted-sum
test__op__weighted__sum_EXTERNAL_OBJECTS =

test/test-op-weighted-sum: CMakeFiles/test-op-weighted-sum.dir/test/test-op-weighted-sum.cc.o
test/test-op-weighted-sum: CMakeFiles/test-op-weighted-sum.dir/build.make
test/test-op-weighted-sum: lib/libgtest.a
test/test-op-weighted-sum: lib/liblnn.a
test/test-op-weighted-sum: lib/openblas.lib
test/test-op-weighted-sum: CMakeFiles/test-op-weighted-sum.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/volume/light-nn/build1/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable test/test-op-weighted-sum"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test-op-weighted-sum.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test-op-weighted-sum.dir/build: test/test-op-weighted-sum

.PHONY : CMakeFiles/test-op-weighted-sum.dir/build

CMakeFiles/test-op-weighted-sum.dir/requires: CMakeFiles/test-op-weighted-sum.dir/test/test-op-weighted-sum.cc.o.requires

.PHONY : CMakeFiles/test-op-weighted-sum.dir/requires

CMakeFiles/test-op-weighted-sum.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test-op-weighted-sum.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test-op-weighted-sum.dir/clean

CMakeFiles/test-op-weighted-sum.dir/depend:
	cd /volume/light-nn/build1 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /volume/light-nn /volume/light-nn /volume/light-nn/build1 /volume/light-nn/build1 /volume/light-nn/build1/CMakeFiles/test-op-weighted-sum.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/test-op-weighted-sum.dir/depend

