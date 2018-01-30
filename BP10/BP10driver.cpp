#include <stdio.h>
#include <stdlib.h>
#include "occa.hpp"
#include <math.h>

// make

// to scan through polynomial degree N
// for N in `seq 1 15`; do ./Ax 4096 $N; done

// type of variables
//KS: started Oct 29th 2017
#if 1
#define datafloat double
#define datafloatString "double"
#else
#define datafloat float
#define datafloatString "float"
#endif
// polynomial degree
#define p_GWJID 0

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

  // number of nodes in layer
  int p_gjNq2  = ((p_N+2)*(p_N+2));
  int BLK = 1;
  int BSIZE  = (p_gjNq*p_gjNq*p_gjNq);


  int p_Nq = (p_N+1);
  int p_Nq2 = p_Nq*p_Nq;
  // number of geometric factors
  int Ngeo =  1;
  int halfD = (p_gjNq+1)/2;
//int halfD =  (p_gjNq+p_gjNq%2)/2;  

int pad;
  // tweak shared padding to avoid bank conflicts
  if ((p_gjNq==8) || (p_gjNq==16))
    pad = 1;
  else
    pad = 0;

  if ((p_gjNq%2)==0)
    pad = 1;
  else
    pad = 0;

  printf("E=%d, N=%d, Ngll=%d\n", E, p_N, p_gjNq);

  // build some dummy storage & parameters
  datafloat *geo, *u, *Mu,*Mu0, *D,*I, *Mutemp, *Mutemp2, *Mutemp3;
  double results2D[8];


  occa::device device;
  occa::kernel BP1kernel[8];
  occa::kernel correctRes;

  occa::memory o_geo, o_u, o_Mu, o_D,o_I,  o_Mutemp, o_Mutemp2,o_Mutemp3;
  occa::kernelInfo kernelInfo;

  //device.setup("mode = Serial");
  // device.setup("mode = OpenMP  , schedule = compact, chunk = 10");
  //device.setup("mode = OpenCL  , platformID = 0, deviceID = 0");
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

  kernelInfo.addDefine("p_halfI", halfD);

  printf("BSIZE=%d\n", BSIZE);



  char buf[200];
  for (int i =1; i<9; i++)
  {
    printf("compiling 2D kernel %d ...\n", i);
    sprintf(buf, "massPartialAxHex3D_Ref%d", i); // puts string into buffer
    BP1kernel[i-1] = device.buildKernelFromSource("massAxHex3D.okl", buf, kernelInfo);
  }


  // initialize with random numbers on Host and Device
  srand48(12345);
  randCalloc(device, E*BSIZE*Ngeo, &geo, o_geo);
  randCalloc(device, E*BSIZE, &u, o_u);
  Mu = (datafloat*) calloc(BSIZE*E , sizeof(datafloat)); 
  o_Mu = device.malloc(BSIZE*E*sizeof(datafloat), Mu);

   randCalloc(device, E*BSIZE, &Mu, o_Mu);
  randCalloc(device, E*BSIZE, &Mutemp, o_Mutemp);
  randCalloc(device, E*BSIZE, &Mutemp2, o_Mutemp2);
//  Mutemp = (datafloat*) calloc(E*BSIZE , sizeof(datafloat));  
 // o_Mutemp = device.malloc(E*BSIZE*sizeof(datafloat), Mutemp);
 // Mutemp2 = (datafloat*) calloc(E*BSIZE , sizeof(datafloat));  
 // o_Mutemp2 = device.malloc(E*BSIZE*sizeof(datafloat), Mutemp2);
//Mutemp3 = (datafloat*) calloc(E*BSIZE , sizeof(datafloat));  
 // o_Mutemp3 = device.malloc(E*BSIZE*sizeof(datafloat), Mutemp3);
  
randCalloc(device, E*BSIZE, &Mutemp3, o_Mutemp3);
//printf("halfD =%d p_gjNq = %d \n", halfD, p_gjNq);

  I = (datafloat*) calloc(p_gjNq*p_Nq , sizeof(datafloat));  
  for(int j=0;j<halfD;++j){
    for (int n=0; n<p_Nq; ++n){
      datafloat tmp = drand48()-0.5;
//datafloat tmp = 1.0f;
      I[j*p_Nq+n] = tmp; 
      I[(p_gjNq-j-1)*p_Nq + (p_Nq-n-1)] =  tmp;
    }
  }


  o_I = device.malloc(p_gjNq*p_Nq*sizeof(datafloat), I);
  // initialize timer
  occa::initTimer(device);

  // queue Ax kernel
  int Niter = 10, it;
double gflops =
 p_gjNq*p_Nq*p_Nq*p_Nq*2 + 
      p_gjNq*p_gjNq*p_Nq*p_Nq*2 + 
      p_gjNq*p_gjNq*p_gjNq*p_Nq*2 +
      p_gjNq*p_gjNq*p_gjNq +
      p_gjNq*p_gjNq*p_gjNq*p_Nq*2 +
      p_gjNq*p_gjNq*p_Nq*p_Nq*2 +
      p_gjNq*p_Nq*p_Nq*p_Nq*2;  
gflops *= Niter;



 
    //device.finish();


  for (int i =1;i<9; i++){
    datafloat lambda = 1.;
   occa::streamTag startTag = device.tagStream();
    for(it=0;it<Niter;++it){

  
    BP1kernel[i-1](E,o_geo,o_I, o_u, o_Mu,o_Mutemp, o_Mutemp2, o_Mutemp3);

    }
    occa::streamTag stopTag = device.tagStream();
    double elapsed = device.timeBetween(startTag, stopTag);

    //device.finish();


    printf("\n\nKERNEL %d  ================================================== \n\n", i);

    printf("OCCA elapsed time = %g\n", elapsed);

    //    double gflops = Niter* (12*p_gjNq*p_gjNq*p_gjNq*p_gjNq + 15*p_gjNq*p_gjNq*p_gjNq);
    results2D[i-1] = E*gflops/(elapsed*1000*1000*1000);
    printf("OCCA: estimated gflops = %17.15f\n", results2D[i-1]);

    printf("OCCA estimated bandwidth = %17.15f GB/s\n", sizeof(datafloat)*Niter*E*(2.*p_Nq*p_Nq*p_Nq + 1.*p_gjNq*p_gjNq*p_gjNq)/(elapsed*1024.*1024.*1024));


    // compute l2 of data
    o_Mu.copyTo(Mu);
    datafloat normMu = 0;
    int n;
    for(n=0;n<E*p_Nq*p_Nq*p_Nq;++n)
      normMu += Mu[n]*Mu[n];
    normMu = sqrt(normMu);

    printf("OCCA: normMu = %17.15lf\n", normMu);

    







//clean the memory (important)
    for(n=0;n<E*BSIZE;++n){
      Mu[n] = 0.0f;
      Mutemp[n] = 0.0f;
      Mutemp2[n] = 0.0f;

    }
    o_Mu.copyFrom(Mu);
    o_Mutemp.copyFrom(Mutemp);
    o_Mutemp2.copyFrom(Mutemp2);

  }

  printf("\n\n ****** ****** *******  SUMMARY ****** ****** *******  \n\n");
  printf("N = %d, Number of el. = %d \n", p_N, E);
  printf("\n");
  printf("Kernels with 2D thread structure (est. gflops) ===> \n"); 
  printf("\n");

  for (int i=0;i<8; ++i){
    printf(" %16.17f ", results2D[i] );
  }
printf("\n");
  exit(0);
  return 0;

}
