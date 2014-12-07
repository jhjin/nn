#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialFlattenedConvolution.c"
#else

static void nn_(conv_1d)(real *y, real *x, real *w, int iC, int iH, int iW, int kL, int horizontal)
{
  long i, j, k;

  // horizontal convolution
  if (horizontal) {
    int oH = iH;
    int oW = iW - kL + 1;
    for(i = 0; i < iC; i++) {
      for(j = 0; j < iH; j++) {
        for(k = 0; k < kL; k++) {
          THVector_(add)(y+i*oH*oW+j*oW, x+i*iH*iW+j*iW+k, *(w+i*kL+k), oW);
        }
      }
    }

  // vertical convolution
  } else {
    int oH = iH - kL + 1;
    int oW = iW;
    for(i = 0; i < iC; i++) {
      for(k = 0; k < kL; k++) {
        THVector_(add)(y+i*oH*oW, x+i*iH*iW+k*iW, *(w+i*kL+k), oH*oW);
      }
    }
  }

  return;
}

static void nn_(SpatialFlattenedConvolution_updateOutput_frame2)(THTensor *input, THTensor *output,
                                                                 THTensor *intm1, THTensor *intm2,
                                                                 THTensor *weight_l, THTensor *weight_v, THTensor *weight_h,
                                                                 THTensor *bias_l, THTensor *bias_v, THTensor *bias_h,
                                                                 int kW, int kH, int dW, int dH, int padding,
                                                                 long nInputPlane, long inputWidth, long inputHeight,
                                                                 long nOutputPlane, long outputWidth, long outputHeight)
{
  long i, j, k;
  THTensor *input2d, *output2d;

  input2d = THTensor_(newWithStorage2d)(input->storage, input->storageOffset,
                                        nInputPlane, -1, inputHeight*inputWidth, -1);
  output2d = THTensor_(newWithStorage2d)(output->storage, output->storageOffset,
                                         nOutputPlane, -1, outputHeight*outputWidth, -1);


  // fill biases
  for(i = 0; i < nOutputPlane; i++) {
    THVector_(fill)(intm1->storage->data+intm1->storageOffset+intm1->stride[0]*i,
                    THTensor_(get1d)(bias_l, i), inputHeight*inputWidth);
    THVector_(fill)(intm2->storage->data+intm2->storageOffset+intm2->stride[0]*i,
                    THTensor_(get1d)(bias_v, i), outputHeight*inputWidth);
    THVector_(fill)(output->storage->data+output->storageOffset+output->stride[0]*i,
                    THTensor_(get1d)(bias_h, i), outputHeight*outputWidth);
  }


  // Lateral convolution
  THTensor_(addmm)(intm1, 1, intm1, 1, weight_l, input2d);

  // Vertical convolution
  nn_(conv_1d)(THTensor_(data)(intm2), THTensor_(data)(intm1), THTensor_(data)(weight_v),
               nOutputPlane, inputHeight, inputWidth, kW, 0);

  // Horizontal convolution
  nn_(conv_1d)(THTensor_(data)(output), THTensor_(data)(intm2), THTensor_(data)(weight_h),
               nOutputPlane, outputHeight, inputWidth, kH, 1);


  THTensor_(free)(input2d);
  THTensor_(free)(output2d);
}

static int nn_(SpatialFlattenedConvolution_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int padding = luaT_getfieldcheckint(L, 1, "padding");

  THTensor *weight_l = luaT_getfieldcheckudata(L, 1, "weight_l", torch_Tensor);
  THTensor *weight_v = luaT_getfieldcheckudata(L, 1, "weight_v", torch_Tensor);
  THTensor *weight_h = luaT_getfieldcheckudata(L, 1, "weight_h", torch_Tensor);
  THTensor *bias_l = luaT_getfieldcheckudata(L, 1, "bias_l", torch_Tensor);
  THTensor *bias_v = luaT_getfieldcheckudata(L, 1, "bias_v", torch_Tensor);
  THTensor *bias_h = luaT_getfieldcheckudata(L, 1, "bias_h", torch_Tensor);
  THTensor *intm1 = luaT_getfieldcheckudata(L, 1, "intm1", torch_Tensor);
  THTensor *intm2 = luaT_getfieldcheckudata(L, 1, "intm2", torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);

  int dimf = 0;
  int dimw = 2;
  int dimh = 1;

  long nInputPlane;
  long inputWidth;
  long inputHeight;
  long nOutputPlane;
  long outputWidth;
  long outputHeight;

  luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D(batch mode) tensor expected");

  // Temporary:
  luaL_argcheck(L, dW == 1, 1, "dW must == 1");
  luaL_argcheck(L, dH == 1, 1, "dH must == 1");

  if (input->nDimension == 4) {
    dimf++;
    dimw++;
    dimh++;
  }

  nInputPlane = input->size[dimf];
  inputWidth   = input->size[dimw];
  inputHeight  = input->size[dimh];
  nOutputPlane = weight_l->size[0];
  outputWidth  = (inputWidth + 2*padding - kW) / dW + 1;
  outputHeight = (inputHeight + 2*padding - kH) / dH + 1;

  if(input->nDimension == 3)
  {
    THTensor_(resize2d)(intm1, nOutputPlane, inputHeight*inputWidth);
    THTensor_(resize2d)(intm2, nOutputPlane, outputHeight*inputWidth);
    THTensor_(resize3d)(output, nOutputPlane, outputHeight, outputWidth);

    nn_(SpatialFlattenedConvolution_updateOutput_frame2)(input, output, intm1, intm2,
                                                        weight_l, weight_v, weight_h,
                                                        bias_l, bias_v, bias_h,
                                                        kW, kH, dW, dH, padding,
                                                        nInputPlane, inputWidth, inputHeight,
                                                        nOutputPlane, outputWidth, outputHeight);
  }
  else
  {
    long T = input->size[0];
    long t;

    THTensor_(resize3d)(intm1, T, nOutputPlane, inputHeight*inputWidth);
    THTensor_(resize3d)(intm2, T, nOutputPlane, outputHeight*inputWidth);
    THTensor_(resize4d)(output, T, nOutputPlane, outputHeight, outputWidth);

    THStorage_(clearFlag)(input->storage, TH_STORAGE_REFCOUNTED);
    THStorage_(clearFlag)(output->storage, TH_STORAGE_REFCOUNTED);
    THStorage_(clearFlag)(intm1->storage, TH_STORAGE_REFCOUNTED);
    THStorage_(clearFlag)(intm2->storage, TH_STORAGE_REFCOUNTED);
    THStorage_(clearFlag)(weight_l->storage, TH_STORAGE_REFCOUNTED);
    THStorage_(clearFlag)(weight_v->storage, TH_STORAGE_REFCOUNTED);
    THStorage_(clearFlag)(weight_h->storage, TH_STORAGE_REFCOUNTED);

#pragma omp parallel for private(t)
    for(t = 0; t < T; t++)
    {
      THTensor *input_t = THTensor_(newSelect)(input, 0, t);
      THTensor *output_t = THTensor_(newSelect)(output, 0, t);
      THTensor *intm1_t = THTensor_(newSelect)(intm1, 0, t);
      THTensor *intm2_t = THTensor_(newSelect)(intm2, 0, t);

      nn_(SpatialFlattenedConvolution_updateOutput_frame2)(input_t, output_t,
                                                          intm1_t, intm2_t,
                                                          weight_l, weight_v, weight_h,
                                                          bias_l, bias_v, bias_h,
                                                          kW, kH, dW, dH, padding,
                                                          nInputPlane, inputWidth, inputHeight,
                                                          nOutputPlane, outputWidth, outputHeight);

      THTensor_(free)(input_t);
      THTensor_(free)(output_t);
      THTensor_(free)(intm1_t);
      THTensor_(free)(intm2_t);
    }
    THStorage_(setFlag)(input->storage, TH_STORAGE_REFCOUNTED);
    THStorage_(setFlag)(output->storage, TH_STORAGE_REFCOUNTED);
    THStorage_(setFlag)(intm1->storage, TH_STORAGE_REFCOUNTED);
    THStorage_(setFlag)(intm2->storage, TH_STORAGE_REFCOUNTED);
    THStorage_(setFlag)(weight_l->storage, TH_STORAGE_REFCOUNTED);
    THStorage_(setFlag)(weight_v->storage, TH_STORAGE_REFCOUNTED);
    THStorage_(setFlag)(weight_h->storage, TH_STORAGE_REFCOUNTED);
  }

  return 1;
}

static const struct luaL_Reg nn_(SpatialFlattenedConvolution__) [] = {
  {"SpatialFlattenedConvolution_updateOutput", nn_(SpatialFlattenedConvolution_updateOutput)},
  {NULL, NULL}
};

static void nn_(SpatialFlattenedConvolution_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(SpatialFlattenedConvolution__), "nn");
  lua_pop(L,1);
}

#endif
