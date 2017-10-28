How to install
1. download code
git clone https://github.com/opencv/opencv.git
git checkout -b v3.3.0
cd opencv
mkdir build
cd build
copy this file to opencv 3.3 version source directory
2. making make file in bulid folder
 
opencv cmake
 cmake         -DCMAKE_BUILD_TYPE=Release         -DCMAKE_INSTALL_PREFIX=/usr/local         -DBUILD_PNG=OFF         -DBUILD_TIFF=OFF         -DBUILD_TBB=OFF         -DBUILD_JPEG=OFF         -DBUILD_JASPER=OFF         -DBUILD_ZLIB=OFF         -DBUILD_EXAMPLES=ON         -DBUILD_opencv_java=OFF         -DBUILD_opencv_python2=ON         -DBUILD_opencv_python3=OFF         -DENABLE_PRECOMPILED_HEADERS=OFF         -DWITH_OPENCL=OFF         -DWITH_FFMPEG=ON         -DWITH_GSTREAMER=OFF         -DWITH_GSTREAMER_0_10=OFF         -DWITH_CUDA=ON         -DWITH_GTK=ON         -DWITH_VTK=OFF         -DWITH_TBB=ON         -DWITH_1394=OFF         -DWITH_OPENEXR=OFF         -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-8.0         -DCUDA_ARCH_BIN=5.3         -DCUDA_ARCH_PTX=""         -DINSTALL_C_EXAMPLES=ON         -DINSTALL_TESTS=OFF ..

3. build and install 
sudo make -j4 install

