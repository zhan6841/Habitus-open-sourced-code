#!/bin/bash
git clone https://android.googlesource.com/platform/external/libpcap
pushd libpcap; mkdir build; mkdir install; popd

export PCAP_SRC_ROOT=${PWD}/libpcap
export CXXFLAGS="-fPIE -pie"
export NDK=/home/anlan/Android/Sdk/ndk/23.1.7779620
export ABI=arm64-v8a
cd libpcap
cd build
cmake -DCMAKE_INSTALL_PREFIX=$PCAP_SRC_ROOT/install -DCMAKE_TOOLCHAIN_FILE=/home/anlan/Android/Sdk/ndk/23.1.7779620/build/cmake/android.toolchain.cmake -DANDROID_ABI=$ABI ..
make
make install