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
  int p_gjNq = (p_N+2); // number of nodes in each direction

  int p_gjNq2  = ((p_N+2)*(p_N+2));
  int BLK = 1;
  int BSIZE  = (p_gjNq*p_gjNq*p_gjNq);
  datafloat lambda = 1.;


  int p_Nq = (p_N+1);
  int p_Nq2 = p_Nq*p_Nq;
  // number of geometric factors
  int Ngeo =  7;
  int halfD = (p_gjNq+1)/2;
  int pad;

  // tweak shared padding to avoid bank conflicts

  if ((p_gjNq%2)==0)
    pad = 1;
  else
    pad = 0;

  int Niter = 10, it;
  double gflops =
    p_gjNq*p_Nq*p_Nq*p_Nq*2 +
    p_gjNq*p_gjNq*p_Nq*p_Nq*2 +
    p_gjNq*p_gjNq*p_gjNq*p_Nq*2 +
    p_gjNq*p_gjNq*p_gjNq*p_gjNq*6 +
    p_gjNq*p_gjNq*p_gjNq*15 +
    p_gjNq*p_gjNq*p_gjNq*p_gjNq*2 +
    p_gjNq*p_gjNq*p_gjNq*2 +
    p_gjNq*p_gjNq*p_gjNq*p_gjNq*4 +
    p_gjNq*p_gjNq*p_gjNq*2 +
    p_gjNq*p_gjNq*p_gjNq*p_Nq*2 +
    p_gjNq*p_gjNq*p_Nq*p_Nq*2 +
    p_gjNq*p_Nq*p_Nq*p_Nq*2 ;
  gflops *= Niter;


  datafloat *geo, *u, *Au,*Au0, *D,*I, *Autemp, *Autemp2;
  double results2D[13];
  double results3D[9];

  occa::device device;
  occa::kernel BK3kernel[13];
  occa::kernel BK3kernel3D[9];
  occa::kernel correctRes;

  occa::memory o_geo, o_u, o_Au, o_D,o_I,  o_Autemp, o_Autemp2;
  occa::kernelInfo kernelInfo;

  // device.setup("mode = Serial");
  // device.setup("mode = OpenMP  , schedule = compact, chunk = 10");
  // device.setup("mode = OpenCL  , platformID = 0, deviceID = 0");
  device.setup("mode = CUDA    , deviceID = 0");
  // device.setup("mode = Pthreads, threadCount = 4, schedule = compact, pinnedCores = [0, 0, 1, 1]");
  // device.setup("mode = COI     , deviceID = 0");

  kernelInfo.addDefine("datafloat", datafloatString);
  kernelInfo.addDefine("p_N", p_N);
  kernelInfo.addDefine("p_gjNq", p_gjNq);
  kernelInfo.addDefine("p_Nq", p_Nq);
  kernelInfo.addDefine("p_Nq2", p_Nq2);
  kernelInfo.addDefine("p_gjNq2", p_gjNq2);
  kernelInfo.addDefine("BSIZE", (int)BSIZE);
  kernelInfo.addDefine("p_Ngeo", Ngeo);
  kernelInfo.addDefine("p_Np", p_Nq*p_Nq*p_Nq);
  kernelInfo.addDefine("pad", pad);
  kernelInfo.addDefine("p_halfD", halfD);

  char buf[200];
  for (int i =1; i<14; i++){
    printf("compiling 2D kernel %d ...\n", i);

    sprintf(buf, "ellipticAxHex3D_Ref2D%d", i); 
    BK3kernel[i-1] = device.buildKernelFromSource("BK3kernels2D.okl", buf, kernelInfo);
  }

  if (p_gjNq<11){
    for (int i =1; i<10; i++){

      printf("compiling 3D kernel %d ...\n", i);

      sprintf(buf, "ellipticAxHex3D_Ref3D%d", i);
      BK3kernel3D[i-1] = device.buildKernelFromSource("BK3kernels3D.okl", buf, kernelInfo);
    }

    printf("compiling 3D control kernel  ...\n");
    correctRes  = device.buildKernelFromSource("BK3kernels3D.okl", "ellipticAxHex3D_e9", kernelInfo);
  }

  srand48(12345);

  randCalloc(device, E*BSIZE*Ngeo, &geo, o_geo);
  randCalloc(device, E*BSIZE, &u, o_u);
  Au = (datafloat*) calloc(BSIZE*E , sizeof(datafloat)); 
  o_Au = device.malloc(BSIZE*E*sizeof(datafloat), Au);

  randCalloc(device, E*BSIZE, &Autemp, o_Autemp);
  randCalloc(device, E*BSIZE, &Autemp2, o_Autemp2);

  randCalloc(device, p_gjNq*p_gjNq, &D, o_D);

  //For correct results, we must have I_{nm} = I_{p_gjNq-n+1, p_Nq-m+1}

  I = (datafloat*) calloc(p_gjNq*p_Nq , sizeof(datafloat));  

  for(int j=0;j<halfD;++j){
    for (int n=0; n<p_Nq; ++n){

      datafloat tmp = drand48()-0.5;

      I[j*p_Nq+n] = tmp; 
      I[(p_gjNq-j-1)*p_Nq + (p_Nq-n-1)] =  tmp;
    }
  }


  o_I = device.malloc(p_gjNq*p_Nq*sizeof(datafloat), I);

  // initialize timer
  occa::initTimer(device);

  for (int i =1;i<14; i++){

    occa::streamTag startTag = device.tagStream();

    device.finish();

    for(it=0;it<Niter;++it){
      BK3kernel[i-1](E,o_geo, o_D,o_I, lambda, o_u, o_Autemp,o_Au );
    }

    occa::streamTag stopTag = device.tagStream();
    device.finish();

    double elapsed = device.timeBetween(startTag, stopTag);

    if (i<=9)
      printf("\n\nKERNEL %d  ================================================== \n\n", i);
    else
      printf("\n\n (bonus) KERNEL %d  ================================================== \n\n", i);

    printf("OCCA elapsed time = %g\n", elapsed);

    results2D[i-1] = E*gflops/(elapsed*1000*1000*1000);

    printf("OCCA: estimated gflops = %17.15f\n", results2D[i-1]);
    printf("OCCA estimated bandwidth = %17.15f GB/s\n", sizeof(datafloat)*Niter*E*(7.*p_gjNq*p_gjNq*p_gjNq+2.0*p_Nq*p_Nq*p_Nq)/(elapsed*1000.*1000.*1000));

    o_Au.copyTo(Au);
    datafloat normAu = 0;

    for(int n=0;n<E*p_Nq*p_Nq*p_Nq;++n)
      normAu += Au[n]*Au[n];

    normAu = sqrt(normAu);

    printf("OCCA: normAu = %17.15lf\n", normAu);

    for(int n=0;n<E*BSIZE;++n){
      Au[n] = 0.0f;

    }
    o_Au.copyFrom(Au);

  }

  double elapsed = 0.0f;
  if (p_gjNq<11){

    for (int i =1;i<10; i++){

      elapsed = 0.0f;

      for(it=0;it<Niter;++it){
        for(int n=0;n<E*BSIZE;++n)
          Au[n] = 0.0f;

        o_Au.copyFrom(Au);
        o_Autemp.copyFrom(Autemp);

        if ((i >1)&&(i<7))
          o_Autemp2.copyFrom(Autemp2);

        device.finish();

        occa::streamTag startTag = device.tagStream();

        if ((i >1)&&(i<7))     
          BK3kernel3D[i-1](E,o_geo, o_D,o_I, lambda, o_u, o_Autemp,o_Autemp2, o_Au );
        else            
          BK3kernel3D[i-1](E,o_geo, o_D,o_I, lambda, o_u, o_Autemp, o_Au);

        occa::streamTag stopTag = device.tagStream();

        elapsed += device.timeBetween(startTag, stopTag);

        device.finish();

      }



      printf("\n\n3D KERNEL %d  ================================================== \n\n", i);
      printf("OCCA elapsed time = %g\n", elapsed);

      results3D[i-1] = E*gflops/(elapsed*1000*1000*1000);

      printf("OCCA: estimated gflops = %17.15f\n", results3D[i-1]);
      printf("OCCA estimated bandwidth = %17.15f GB/s\n", sizeof(datafloat)*Niter*E*(7.*p_gjNq*p_gjNq*p_gjNq+2.0*p_Nq*p_Nq*p_Nq)/(elapsed*1000.*1000.*1000));

      // compute l2 of data

      o_Au.copyTo(Au);
      datafloat normAu = 0.0f;

      for(int n=0;n<E*p_Nq*p_Nq*p_Nq;++n)
        normAu += Au[n]*Au[n];

      normAu = sqrt(normAu);

      printf("OCCA: normAu = %17.15lf\n", normAu);
    }

    for(int n=0;n<E*BSIZE;++n){
      Au[n] = 0.0f;
    }

    o_Au.copyFrom(Au);
    o_Autemp.copyFrom(Autemp);

    if (p_gjNq<11){  

      correctRes(E, o_geo, o_D,o_I, lambda, o_u, o_Autemp, o_Au);

      o_Au.copyTo(Au);

      datafloat normAu = 0;

      for(int n=0;n<E*p_Nq*p_Nq*p_Nq;++n)
        normAu += Au[n]*Au[n];

      normAu = sqrt(normAu);

      printf("\n RESULT SHOULD BE: normAu = %17.15lf\n", normAu); 
    }
    printf("\n");
  }

  printf("\n\n ******** ****** ******  SUMMARY ****** ****** ******** \n\n");
  printf("N = %d, Number of el. = %d \n", p_N, E);
  printf("\n");
  printf("Kernels with 2D thread structure (est. gflops) ===> \n"); 
  printf("\n");
printf("data%d(%d, :) = [ ", E, p_N);
  for (int i=0;i<13; ++i){
    printf(" %16.17f ", results2D[i] );
  }
printf("];");
  printf("\n\n");
  if (p_gjNq<11){

    printf("Kernels with 3D thread structure (est. gflops) ===> \n");
    printf("\n");
printf("data3D%d(%d, :) = [ ", E, p_N);
    for (int i=0;i<9; ++i){ 
      printf(" %16.17f ", results3D[i] );
    }
printf("];");
  }

  printf("\n");

  exit(0);
  return 0;
}
