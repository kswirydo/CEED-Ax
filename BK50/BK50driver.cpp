#include <stdio.h>
#include <stdlib.h>
#include "occa.hpp"
#include <math.h>

#if 1
#define datafloat double
#define datafloatString "double"
#else
#define datafloat float
#define datafloatString "float"
#endif

void randCalloc(occa::device &device, int sz, datafloat **pt, occa::memory &o_pt){

  *pt = (datafloat*) calloc(sz, sizeof(datafloat));
  int n;
  datafloat sum = 0;
  for(n=0;n<sz;++n){
    (*pt)[n] = drand48()-0.5;
    sum += pow((*pt)[n],2);
  }

  o_pt = device.malloc(sz*sizeof(datafloat), *pt);
}

int main(int argc, char **argv){

  // default to 512 elements if no arg is given
  int E = (argc>=2) ? atoi(argv[1]):512;
  int p_N = (argc>=3) ? atoi(argv[2]):15;
  int p_Ngll = (p_N+1); // number of nodes in each direction

  int p_Ngll2  = ((p_N+1)*(p_N+1));
  int BLK = 1;
  int BSIZE  = (p_Ngll*p_Ngll*p_Ngll);

  // number of geometric factors
  int Ngeo =  7;

  int pad;

  // tweak shared padding to avoid bank conflicts
  if ((p_Ngll%2)==0)
    pad = 1;
  else
    pad = 0;

  datafloat lambda = 1.;
  datafloat *geo, *u, *Au, *D, *Autemp;

  occa::device device;
  occa::kernel BK50kernel[10];
  occa::kernel BK50kernel3D[6];
  occa::memory o_geo, o_u, o_Au, o_D, o_Autemp;
  occa::kernelInfo kernelInfo;

  double results2D[10];
  double  results3D[6];

  // device.setup("mode = Serial");
  // device.setup("mode = OpenMP  , schedule = compact, chunk = 10");
  //  device.setup("mode = OpenCL  , platformID = 0, deviceID = 1");
  device.setup("mode = CUDA    , deviceID = 0");
  // device.setup("mode = Pthreads, threadCount = 4, schedule = compact, pinnedCores = [0, 0, 1, 1]");
  // device.setup("mode = COI     , deviceID = 0");

  kernelInfo.addDefine("datafloat", datafloatString);
  kernelInfo.addDefine("p_N", p_N);
  kernelInfo.addDefine("p_Ngll", p_Ngll);
  kernelInfo.addDefine("p_Ngll2", p_Ngll2);
  kernelInfo.addDefine("BSIZE", (int)BSIZE);
  kernelInfo.addDefine("p_Ngeo", Ngeo);
  kernelInfo.addDefine("p_Np", BSIZE);
  kernelInfo.addDefine("pad", pad);

  char buf[200];
  for (int i =1; i<11; i++){

    sprintf(buf, "ellipticAxHex3D_Ref2D%d", i);
    BK50kernel[i-1] = device.buildKernelFromSource("BK50kernels2D.okl", buf, kernelInfo);
  }

  if (p_Ngll<11){
    for (int i =1; i<7; i++){

      sprintf(buf, "ellipticAxHex3D_Ref3D%d", i);
      BK50kernel3D[i-1] = device.buildKernelFromSource("BK50kernels3D.okl", buf, kernelInfo);
    }
  }

  // initialize with random numbers on Host and Device
  srand48(12345);

  randCalloc(device, E*BSIZE*Ngeo, &geo, o_geo);
  randCalloc(device, E*BSIZE, &u, o_u);
  randCalloc(device, E*BSIZE, &Au, o_Au);
  randCalloc(device, E*BSIZE, &Autemp, o_Autemp);
  randCalloc(device, p_Ngll*p_Ngll, &D, o_D);

  // initialize timer
  occa::initTimer(device);

  int Niter = 10, it;

  double gflops = Niter* (12*p_Ngll*p_Ngll*p_Ngll*p_Ngll + 15*p_Ngll*p_Ngll*p_Ngll);

  for (int i =1;i<11; i++){

    device.finish();

    occa::streamTag startTag = device.tagStream();
    for(it=0;it<Niter;++it){
      BK50kernel[i-1](E,o_geo, o_D, lambda, o_u, o_Au, o_Autemp);
    }

    occa::streamTag stopTag = device.tagStream();

    double elapsed = device.timeBetween(startTag, stopTag);
    printf("\n\nKERNEL %d  ================================================== \n\n", i);
    printf("OCCA elapsed time = %g\n", elapsed);

    results2D[i-1] = E*gflops/(elapsed*1000*1000*1000);
    printf("OCCA: estimated gflops = %17.15f\n", results2D[i-1]);
    printf("OCCA estimated bandwidth = %17.15f GB/s\n", sizeof(datafloat)*Niter*E*9.*(p_Ngll*p_Ngll*p_Ngll)/(elapsed*1000.*1000.*1000));

    // compute l2 of data
    o_Au.copyTo(Au);

    datafloat normAu = 0;
    for(int n=0;n<E*BSIZE;++n)
      normAu += Au[n]*Au[n];

    normAu = sqrt(normAu);
    printf("OCCA: normAu = %17.15lf\n", normAu);
  }

  if (p_Ngll<11){
    for (int i =1;i<7; i++){

      device.finish();

      occa::streamTag startTag = device.tagStream();

      for(it=0;it<Niter;++it){
        BK50kernel3D[i-1](E,o_geo, o_D, lambda, o_u, o_Au, o_Autemp);
      }

      occa::streamTag stopTag = device.tagStream();

      double elapsed = device.timeBetween(startTag, stopTag);
      printf("\n\n3D KERNEL %d  ================================================== \n\n", i);

      printf("OCCA elapsed time = %g\n", elapsed);

      results3D[i-1] = E*gflops/(elapsed*1000*1000*1000);
      printf("OCCA: estimated gflops = %17.15f\n", results3D[i-1]);
      printf("OCCA estimated bandwidth = %17.15f GB/s\n", sizeof(datafloat)*Niter*E*9.*(p_Ngll*p_Ngll*p_Ngll)/(elapsed*1000.*1000.*1000));

      // compute l2 of data
      o_Au.copyTo(Au);

      datafloat normAu = 0;
      for(int n=0;n<E*BSIZE;++n)
        normAu += Au[n]*Au[n];

      normAu = sqrt(normAu);
      printf("OCCA: normAu = %17.15lf\n", normAu);
    }
  }

  printf("\n\n ****** ****** *******  SUMMARY ****** ****** ******* \n\n");
  printf("N = %d, Number of el. = %d \n", p_N, E);
  printf("\n");
  printf("Kernels with 2D thread structure (est. gflops) ===> \n");
  printf("\n");

  for (int i=0; i<10; ++i){
    printf(" %16.17f ", results2D[i]);
  }
  printf("\n\n");

  if (p_Ngll<11){

    printf("Kernels with 3D thread structure (est. gflops) ===> \n");
    printf("\n");

    for (int i=0; i<6; ++i){
      printf(" %16.17f ", results3D[i]);
    }
    printf("\n\n");
  }

  exit(0);
  return 0;
}
