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
CMAKE_SOURCE_DIR = /pycharm-projects/light-nn

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /pycharm-projects/light-nn/build1

# Include any dependencies generated for this target.
include CMakeFiles/test-op-reverse.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/test-op-reverse.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test-op-reverse.dir/flags.make

CMakeFiles/test-op-reverse.dir/test/test-op-reverse.cc.o: CMakeFiles/test-op-reverse.dir/flags.make
CMakeFiles/test-op-reverse.dir/test/test-op-reverse.cc.o: ../test/test-op-reverse.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/pycharm-projects/light-nn/build1/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/test-op-reverse.dir/test/test-op-reverse.cc.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test-op-reverse.dir/test/test-op-reverse.cc.o -c /pycharm-projects/light-nn/test/test-op-reverse.cc

CMakeFiles/test-op-reverse.dir/test/test-op-reverse.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test-op-reverse.dir/test/test-op-reverse.cc.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /pycharm-projects/light-nn/test/test-op-reverse.cc > CMakeFiles/test-op-reverse.dir/test/test-op-reverse.cc.i

CMakeFiles/test-op-reverse.dir/test/test-op-reverse.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test-op-reverse.dir/test/test-op-reverse.cc.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /pycharm-projects/light-nn/test/test-op-reverse.cc -o CMakeFiles/test-op-reverse.dir/test/test-op-reverse.cc.s

CMakeFiles/test-op-reverse.dir/test/test-op-reverse.cc.o.requires:

.PHONY : CMakeFiles/test-op-reverse.dir/test/test-op-reverse.cc.o.requires

CMakeFiles/test-op-reverse.dir/test/test-op-reverse.cc.o.provides: CMakeFiles/test-op-reverse.dir/test/test-op-reverse.cc.o.requires
	$(MAKE) -f CMakeFiles/test-op-reverse.dir/build.make CMakeFiles/test-op-reverse.dir/test/test-op-reverse.cc.o.provides.build
.PHONY : CMakeFiles/test-op-reverse.dir/test/test-op-reverse.cc.o.provides

CMakeFiles/test-op-reverse.dir/test/test-op-reverse.cc.o.provides.build: CMakeFiles/test-op-reverse.dir/test/test-op-reverse.cc.o


# Object files for target test-op-reverse
test__op__reverse_OBJECTS = \
"CMakeFiles/test-op-reverse.dir/test/test-op-reverse.cc.o"

# External object files for target test-op-reverse
test__op__reverse_EXTERNAL_OBJECTS =

test/test-op-reverse: CMakeFiles/test-op-reverse.dir/test/test-op-reverse.cc.o
test/test-op-reverse: CMakeFiles/test-op-reverse.dir/build.make
test/test-op-reverse: lib/libgtest.a
test/test-op-reverse: lib/liblnn.a
test/test-op-reverse: lib/libopenblas.a
test/test-op-reverse: CMakeFiles/test-op-reverse.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/pycharm-projects/light-nn/build1/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable test/test-op-reverse"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test-op-reverse.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test-op-reverse.dir/build: test/test-op-reverse

.PHONY : CMakeFiles/test-op-reverse.dir/build

CMakeFiles/test-op-reverse.dir/requires: CMakeFiles/test-op-reverse.dir/test/test-op-reverse.cc.o.requires

.PHONY : CMakeFiles/test-op-reverse.dir/requires

CMakeFiles/test-op-reverse.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test-op-reverse.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test-op-reverse.dir/clean

CMakeFiles/test-op-reverse.dir/depend:
	cd /pycharm-projects/light-nn/build1 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /pycharm-projects/light-nn /pycharm-projects/light-nn /pycharm-projects/light-nn/build1 /pycharm-projects/light-nn/build1 /pycharm-projects/light-nn/build1/CMakeFiles/test-op-reverse.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/test-op-reverse.dir/depend

