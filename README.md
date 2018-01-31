# CEED-Ax

The code needs OCCA to run ( https://github.com/libocca/occa )

Install OCCA first.

git clone https://github.com/libocca/occa.git

Enter occa directory.

Type make.

Set environmental variables
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${PWD}/lib"
PATH+=":${PWD}/bin"

[set occa_dir]

You can also add the commands to your .bashrc

export OCCA_DIR=/path/to/folder/occa/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$OCCA_DIR/lib

To run the code in the repo:

## BP1.0
cd BP10

make

./BP10 512 10

[to simulate mass-matrix-vector multiplication on a mesh with 512 elements and polynomial degree 10 using 12^3 quadrature nodes]

## BP3.0
cd BP30

make

./BP30 512 10

[to simulate stiffness-matrix-vector multiplication on a mesh with 512 elements and polynomial degree 10 using 12^3 quadrature nodes]

## BP3.5

cd BP35

make

./BP35 512 10

[to simulate stiffness-matrix-vector multiplication on a mesh with 512 elements and polynomial degree 10 using 11^3 quadrature nodes]
