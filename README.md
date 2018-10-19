# CEED-Ax

## Install OCCA

The code needs OCCA to run (https://github.com/libocca/occa  )

Install stable version of OCCA first.

git clone https://github.com/libocca/occa.git -b 0.2

Enter occa directory.

Type make.

Set environmental variables

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${PWD}/lib"

PATH+=":${PWD}/bin"

export OCCA_DIR=/path/to/folder/occa/

You can also add the commands to your .bashrc

export OCCA_DIR=/path/to/folder/occa/

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$OCCA_DIR/lib

To run the code in the repo:

## Running BK1.0
cd BK10

make

./BK10 512 10

[to simulate mass-matrix-vector multiplication on a mesh with 512 elements and polynomial degree 10 using 12^3 quadrature nodes]

## Running BK3.0
cd BK30

make

./BPK0 512 10

[to simulate stiffness-matrix-vector multiplication on a mesh with 512 elements and polynomial degree 10 using 12^3 quadrature nodes]

## Running BK5.0

cd BK50

make

./BK50 512 10

[to simulate stiffness-matrix-vector multiplication on a mesh with 512 elements and polynomial degree 10 using 11^3 quadrature nodes]

## What to expect

 The results vary depending on the hardware. For example, on an NVIDIA Tesla P100 and V100 we expect the results to be close to the roofline based on device-to-device data copy but on Fermi and Kepler class cards the performance is far from the peak.
 Examples:
 [Results on NVIDIA K40c](BK1Fig4096STANDALONEk40.pdf) and
  [Results on NVIDIA V100](BK10V100.pdf)


## Problems?

Email katarzyna.swirydowicz@nrel.gov
