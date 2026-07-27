libcuckoo
=========

libcuckoo provides a high-performance, compact hash table that allows
multiple concurrent reader and writer threads.

The Doxygen-generated documentation is available at the
[project page](http://efficient.github.io/libcuckoo/).

Authors: Manu Goyal, Bin Fan, Xiaozhou Li, David G. Andersen, and Michael Kaminsky

For details about this algorithm and citations, please refer to
our papers in [NSDI 2013][1] and [EuroSys 2014][2]. Some of the details of the hashing
algorithm have been improved since that work (e.g., the previous algorithm
in [1] serializes all writer threads, while our current
implementation supports multiple concurrent writers), however, and this source
code is now the definitive reference.

   [1]: http://www.cs.cmu.edu/~dga/papers/memc3-nsdi2013.pdf "MemC3: Compact and Concurrent Memcache with Dumber Caching and Smarter Hashing"
   [2]: http://www.cs.princeton.edu/~mfreed/docs/cuckoo-eurosys14.pdf "Algorithmic Improvements for Fast Concurrent Cuckoo Hashing"

Requirements
================

This library has been tested on Mac OSX >= 10.8 and Ubuntu >= 12.04.

It compiles with clang++ >= 3.3 and g++ >= 4.8, however we strongly suggest
using the latest versions of both compilers, as they have greatly improved
support for atomic operations. Building the library requires CMake version >=
3.1.0. To install it on Ubuntu

    $ sudo apt-get update && sudo apt-get install cmake

Building
==========

libcuckoo is a header-only library, so in order to get going, just add the
`libcuckoo` subdirectory to your include path. These directions cover
installing the library to a particular location on your system, and also
building any the examples and tests included in the repository.

We suggest you build out of source, in a separate `build` directory:

    $ mkdir build
    $ cd build

There are numerous flags you can pass to `CMake` to set which parts of the
repository it builds.

`-DCMAKE_INSTALL_PREFIX`
: set the location where the libcuckoo header files are installed

`-DCMAKE_BUILD_TYPE`
: enable different types of build flags for different purposes

`-DBUILD_EXAMPLES=1`
: tell `CMake` to build the `examples` directory

`-DBUILD_TESTS=1`
: build all tests in the `tests` directory

`-DBUILD_STRESS_TESTS=1`
: build all tests in the `tests/stress-tests` directory

`-DBUILD_UNIT_TESTS=1`
: build all tests in the `tests/unit-tests` directory

`-DBUILD_UNIVERSAL_BENCHMARK=1`
: build the universal benchmark in the `tests/universal-benchmark` directory.
This benchmark allows you to test a variety of operations in arbitrary
percentages, with specific keys and values.  Consult the `README` in the
benchmark directory for more details.

So, if, for example, we want to build all examples and all tests into a local
installation directory, we'd run the following command from the `build`
directory.

    $ cmake -DCMAKE_INSTALL_PREFIX=../install -DBUILD_EXAMPLES=1 -DBUILD_TESTS=1 ..
    $ make all
    $ make install

Usage
==========

When compiling your own files with `libcuckoo`, always remember to enable C++11
features on your compiler. On `g++`, this would be `-std=c++11`, and on
`clang++`, this would be `-std=c++11 -stdlib=libc++`.

Once you have installed the header files and the install location has been added
to your search path, you can include `<libcuckoo/cuckoohash_map.hh>`, and any of
the other headers you installed, into your source file.

There is also a C wrapper around the table that can be leveraged to use
`libcuckoo` in a C program. The interface consists of a template header and
implementation file that can be used to generate instances of the hashtable for
different key-value types.

See the `examples` directory for a demonstration of all of these features.

Tests
==========

The `tests` directory contains a number of tests and benchmarks of the hash
table, which also can serve as useful examples of how to use the table's various
features. Make sure to enable the tests you want to build with the corresponding
`CMake` flags. The test suite can be run with the `make test` command. The test
executables can be run individually as well.

Issue Report
============

To let us know your questions or issues, we recommend you
[report an issue](https://github.com/efficient/libcuckoo/issues) on
github. You can also email us at
[libcuckoo-dev@googlegroups.com](mailto:libcuckoo-dev@googlegroups.com).

Licence
===========
Copyright (C) 2013, Carnegie Mellon University and Intel Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

---------------------------

The third-party libraries have their own licenses, as detailed in their source
files.
