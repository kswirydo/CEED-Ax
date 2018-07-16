# CEED-Ax

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

## BK1.0
cd BK10

make

./BK10 512 10

[to simulate mass-matrix-vector multiplication on a mesh with 512 elements and polynomial degree 10 using 12^3 quadrature nodes]

## BK3.0
cd BK30

make

./BPK0 512 10

[to simulate stiffness-matrix-vector multiplication on a mesh with 512 elements and polynomial degree 10 using 12^3 quadrature nodes]

## BP5.0

cd BP50

make

./BP50 512 10

[to simulate stiffness-matrix-vector multiplication on a mesh with 512 elements and polynomial degree 10 using 11^3 quadrature nodes]
