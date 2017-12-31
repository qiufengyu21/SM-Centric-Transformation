#ifndef _SMC_HEADER
#define _SMC_HEADER

/* Return the number of workers (e.g., CTAs) per SM */
unsigned int __SMC_numNeeded(){
  return 2; /* to be made more flexible; could be determined by the maximum number of CTAs that can be active */
}

// __SMC_worksNeeded is the number of CTAs needed per SM
// Assuming device 0 is used
// __SMC_worderCount is an array with #SMs elements, all set to 0
#define __SMC_init()\
    int __SMC_totalChunks = __SMC_orgGridDim.x * __SMC_orgGridDim.y; \
 cudaDeviceProp __SMC_deviceProp; \
 cudaGetDeviceProperties(&__SMC_deviceProp, 0); \
 int __SMC_sms = __SMC_deviceProp.multiProcessorCount; \
 int __SMC_workersNeeded = __SMC_numNeeded();\
int * __SMC_newChunkSeq=NULL, * __SMC_seqEnds=NULL;\
 __SMC_buildChunkSeq(__SMC_totalChunks, __SMC_sms, __SMC_workersNeeded, __SMC_newChunkSeq, __SMC_seqEnds); \
 int * __SMC_workerCount= __SMC_initiateArray(__SMC_sms);\
 dim3 grid(__SMC_workersNeeded*__SMC_sms,1);
 //debugging
 //printf("There are %d SMs on this GPU, model number: %d.%d\n",__SMC_sms, __SMC_deviceProp.major, __SMC_deviceProp.minor);


  //degugging code that could be put in __SMC_Begin before its for loop
  //  if (threadIdx.x==0 && threadIdx.y==0){				\
//    printf("thread block (%d,%d) is on SM %d, __SMC_CTAID=%d.\n", blockIdx.x, blockIdx.y, __SMC_smid, __SMC_CTAID); \
//    printf("range of idx is [%d, %d).\n", __SMC_seqEnds[__SMC_CTAID-1], __SMC_seqEnds[__SMC_CTAID]); \
//  }									\

#define __SMC_Begin \
      __shared__ int __SMC_workingCTAs;\
    uint __SMC_smid;\
    asm("mov.u32 %0, %smid;" : "=r"( __SMC_smid) );\
    if(threadIdx.x == 0 && threadIdx.y==0)\
      __SMC_workingCTAs =\
        atomicInc ((unsigned int *)& (__SMC_workerCount[ __SMC_smid]), 1024);\
    __syncthreads();\
    if( __SMC_workingCTAs >= __SMC_workersNeeded) return;\
    int __SMC_CTAID = __SMC_smid*__SMC_workersNeeded+__SMC_workingCTAs+1;\
    for (int __SMC_chunkIDidx = __SMC_seqEnds[__SMC_CTAID-1]; __SMC_chunkIDidx < __SMC_seqEnds[__SMC_CTAID]; __SMC_chunkIDidx++){\
       int __SMC_chunkID = __SMC_newChunkSeq[__SMC_chunkIDidx];

#define __SMC_End }

void __SMC_buildChunkSeq(int totalChunks, int sms, int ctasPerSM, int * & seqs_d, int * & seqEnds_d){
  // build an naive sequence from 0 to totalChunks-1; more sophisticated sequence could be used to replace it.
  int sz = totalChunks*sizeof(int);
  int * seqs_h = (int *)malloc(sz);
  for (int i=0;i<totalChunks;i++)
    seqs_h[i] = i;
  cudaMalloc((void**) &seqs_d, sz);
  cudaMemcpy(seqs_d, seqs_h, sz, cudaMemcpyHostToDevice);

  // build the seqEnds array; seqEnds[i] (i>0) marks the ending chunkID for a CTA.
  // seqEnds[0] is always 0.
  // First, figure out the number of chunks assigned to each SM
  int * chunksPerSM = (int *)malloc(sms*sizeof(int));
  int chunksPerSM_ = totalChunks/sms;
  int remains = totalChunks - chunksPerSM_*sms;
  sz=sizeof(int)*(ctasPerSM*sms+1);
  int * seqEnds_h = (int *)malloc(sz);
  // different SMs could have different numbers of SMs when totalCunks is not a multiple of sms
  for (int i=0; i<sms; i++){
    chunksPerSM[i] = i<remains? chunksPerSM_+1: chunksPerSM_;
  }
  // Second, figure out the numbers of chunks assigned to each CTA
  int m = 0;
  seqEnds_h[m++] = 0; //0 as the starting mark of the chunks for the first SM
  for (int i=0; i<sms; i++){
    int chunksPerCTA_ = chunksPerSM[i]/ctasPerSM;
    int remains = chunksPerSM[i] - chunksPerCTA_*ctasPerSM;
    for (int j=0; j<ctasPerSM; j++){
      seqEnds_h[m] = j<remains? seqEnds_h[m-1]+chunksPerCTA_+1:seqEnds_h[m-1]+chunksPerCTA_;
      m++;
    }
  }
  cudaMalloc((void**) &seqEnds_d, sz);
  cudaMemcpy(seqEnds_d, seqEnds_h, sz, cudaMemcpyHostToDevice);

  // debugging
  // printf("Totally, there are %d end marks for %d CTAs.\n", m, ctasPerSM*sms);
  // printf("seqEnds[0]=%d\n",seqEnds_h[0]);
  // for (int i=1; i<m; i++){
  //   printf("seqEnds[%d]=%d, chunks=%d\n", i, seqEnds_h[i], seqEnds_h[i]-seqEnds_h[i-1]);
  // }
}
			
int * __SMC_buildChunkSeq(int totalChunks){
  int sz = totalChunks*sizeof(int);
  int * seqs_h = (int *)malloc(sz);
  for (int i=0;i<totalChunks;i++)
    seqs_h[i] = i;
  int * seqs_d;
  cudaMalloc((void**) &seqs_d, sz);
  cudaMemcpy(seqs_d, seqs_h, sz, cudaMemcpyHostToDevice);
  return seqs_d;
}

int * __SMC_initiateArray(int sms){
  int * d_wc;
  cudaMalloc((void **)&d_wc, sms*sizeof(int));
  cudaMemset((void *)d_wc, 0, sms*sizeof(int));
  return d_wc;
}

#endif
