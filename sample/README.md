# Sample

`mesh_sampling.cpp` is used to sample point clouds uniformly from CAD model. In order to compile it, you have to install:

* CMake
* PCL
* VTK

## CMake

Use this command to install CMake:

```bash
sudo apt-get udpate
sudo apt-get install cmake
```

## PCL

The version I used is the latest version, you can use these commands to install:

```bash
sudo apt-get update  
sudo apt-get install git build-essential linux-libc-dev
sudo apt-get install cmake cmake-gui
sudo apt-get install libusb-1.0-0-dev libusb-dev libudev-dev
sudo apt-get install mpi-default-dev openmpi-bin openmpi-common 
sudo apt-get install libflann1.9 libflann-dev
sudo apt-get install libeigen3-dev 
sudo apt-get install libboost-all-dev
sudo apt-get install libqhull* libgtest-dev
sudo apt-get install freeglut3-dev pkg-config
sudo apt-get install libxmu-dev libxi-dev
sudo apt-get install mono-complete
sudo apt-get install openjdk-8-jdk openjdk-8-jre

git clone https://github.com/PointCloudLibrary/pcl.git
cd pcl
mkdir build && cd build
cmake ..
make -j4
sudo make install
```

## VTK

The version of the VTK is `8.2.0`. You can download it from the [website](https://vtk.org/download/) and use the commands blew to install:

```bash
tar -xzvf VTK-8.2.0.zip
cd VTK-8.2.0/
```

Before compiling, you need to edit the file `IO/Geometry/vtkOBJReader.cxx`. In line 859, add the following code:

```C++
// Here we turn off texturing and/or normals
if (n_tcoord_pts == 0)
{
    hasTCoords = false;
}
if (n_normal_pts == 0)
{
    hasNormals = false;
}
```

Continue to build:

```bash
mkdir build && cd build
cmake ..
make -j4
sudo make install
```

## Compile

In order to use the script, you need to compile it:

```bash
cd sample
mkdir build && cd build
cmake ..
make
```

And you can get a exectuable file `mesh_sampling` in the `build` directory. You can use `mesh_sampling -h` for help. I've provided the `mesh_sampling`. But there are some problems with the options of the command. The option `-n_samples` seems cannot work.

## Example

CAD model and sampled point cloud :

<img src="../images/cad.png" width="300px"/>

<img src="../images/ground_truth.png" width="300px"/>
