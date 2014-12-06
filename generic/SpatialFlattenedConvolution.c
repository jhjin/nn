#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialFlattenedConvolution.c"
#else

static void nn_(unfolded_copy2)(THTensor *finput, THTensor *input,
                               int kW, int kH,
                               int dW, int dH,
                               int padding,
                               int nInputPlane,
                               int inputWidth, int inputHeight,
                               int outputWidth, int outputHeight)
{
  long k;
  real *input_data = THTensor_(data)(input);
  real *finput_data = THTensor_(data)(finput);

#pragma omp parallel for private(k)
  for(k = 0; k < nInputPlane*kH*kW; k++) {
    int nip = k / (kH*kW);
    int rest = k % (kH*kW);
    int kh = rest / kW;
    int kw = rest % kW;
    int x,y,ix,iy;
    real *dst = finput_data + nip*(kH*kW*outputHeight*outputWidth) + kh*(kW*outputHeight*outputWidth) + kw*(outputHeight*outputWidth);
    real *src = input_data + nip*(inputHeight*inputWidth);
    for(y = 0; y < outputHeight; y++) {
      iy = y + kh;
      ix = 0 + kw;
      memcpy(dst+y*outputWidth, src+iy*inputWidth+ix, sizeof(real)*outputWidth);
    }
  }
}

static void nn_(SpatialFlattenedConvolution_updateOutput_frame)(THTensor *input, THTensor *output,
                                                                THTensor *tmp_l, THTensor *tmp_v,
                                                                THTensor *finput_v, THTensor *finput_h,
                                                                THTensor *weight_l, THTensor *weight_v, THTensor *weight_h,
                                                                THTensor *bias_l, THTensor *bias_v, THTensor *bias_h,
                                                                int kW, int kH, int dW, int dH, int padding,
                                                                long nInputPlane, long inputWidth, long inputHeight,
                                                                long nOutputPlane, long outputWidth, long outputHeight)
{
  long i;
  THTensor *input2d, *output2d;

  input2d = THTensor_(newWithStorage2d)(input->storage, input->storageOffset, nInputPlane, -1, inputHeight*inputWidth, -1);
  output2d = THTensor_(newWithStorage2d)(output->storage, output->storageOffset, nOutputPlane, -1, outputHeight*outputWidth, -1);


  // Lateral convolution
  for(i = 0; i < nOutputPlane; i++)
    THVector_(fill)(tmp_l->storage->data+tmp_l->storageOffset+tmp_l->stride[0]*i, THTensor_(get1d)(bias_l, i), inputHeight*inputWidth);

  THTensor_(addmm)(tmp_l, 1, tmp_l, 1, weight_l, input2d);


  // Vertical convolution
  nn_(unfolded_copy2)(finput_v, tmp_l, 1, kH, 1, dH, padding, nOutputPlane, inputWidth, inputHeight, inputWidth, outputHeight);

  for(i = 0; i < nOutputPlane; i++)
    THVector_(fill)(tmp_v->storage->data+tmp_v->storageOffset+tmp_v->stride[0]*i, THTensor_(get1d)(bias_v, i), outputHeight*inputWidth);

  for(i = 0; i < nOutputPlane; i++) {
    THTensor *a = THTensor_(newWithStorage2d)(weight_v->storage, weight_v->storageOffset+weight_v->stride[0]*i, 1, -1, kH, -1);
    THTensor *b = THTensor_(newWithStorage2d)(finput_v->storage, finput_v->storageOffset+finput_v->stride[0]*i*kH, kH, -1, outputHeight*inputWidth, -1);
    THTensor *c = THTensor_(newWithStorage2d)(tmp_v->storage, tmp_v->storageOffset+tmp_v->stride[0]*i, 1, -1, outputHeight*inputWidth, -1);

    THTensor_(addmm)(c, 1, c, 1, a, b); // 2x15

    THTensor_(free)(a);
    THTensor_(free)(b);
    THTensor_(free)(c);
  }


  // Horizontal convolution
  nn_(unfolded_copy2)(finput_h, tmp_v, kW, 1, dW, 1, padding, nOutputPlane, inputWidth, outputHeight, outputWidth, outputHeight);

  for(i = 0; i < nOutputPlane; i++)
    THVector_(fill)(output->storage->data+output->storageOffset+output->stride[0]*i, THTensor_(get1d)(bias_h, i), outputHeight*outputWidth);

  for(i = 0; i < nOutputPlane; i++) {
    THTensor *a = THTensor_(newWithStorage2d)(weight_h->storage, weight_h->storageOffset+weight_h->stride[0]*i, 1, -1, kW, -1);
    THTensor *b = THTensor_(newWithStorage2d)(finput_h->storage, finput_h->storageOffset+finput_h->stride[0]*i*kW, kH, -1, outputHeight*outputWidth, -1);
    THTensor *c = THTensor_(newWithStorage2d)(output2d->storage, output2d->storageOffset+output2d->stride[0]*i, 1, -1, outputHeight*outputWidth, -1);

    THTensor_(addmm)(c, 1, c, 1, a, b); // 2x15

    THTensor_(free)(a);
    THTensor_(free)(b);
    THTensor_(free)(c);
  }


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
  THTensor *tmp_l = luaT_getfieldcheckudata(L, 1, "tmp_l", torch_Tensor);
  THTensor *tmp_v = luaT_getfieldcheckudata(L, 1, "tmp_v", torch_Tensor);
  THTensor *finput_v = luaT_getfieldcheckudata(L, 1, "finput_v", torch_Tensor);
  THTensor *finput_h = luaT_getfieldcheckudata(L, 1, "finput_h", torch_Tensor);
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
    THTensor_(resize2d)(finput_v, kH*nOutputPlane, outputHeight*inputWidth);
    THTensor_(resize2d)(finput_h, kW*nOutputPlane, outputHeight*outputWidth);
    THTensor_(resize2d)(tmp_l, nOutputPlane, inputHeight*inputWidth);
    THTensor_(resize2d)(tmp_v, nOutputPlane, outputHeight*inputWidth);
    THTensor_(resize3d)(output, nOutputPlane, outputHeight, outputWidth);

    nn_(SpatialFlattenedConvolution_updateOutput_frame)(input, output, tmp_l, tmp_v, finput_v, finput_h,
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

    THTensor_(resize3d)(finput_v, T, kH*nOutputPlane, outputHeight*inputWidth);
    THTensor_(resize3d)(finput_h, T, kW*nOutputPlane, outputHeight*outputWidth);
    THTensor_(resize3d)(tmp_l, T, nOutputPlane, inputHeight*inputWidth);
    THTensor_(resize3d)(tmp_v, T, nOutputPlane, outputHeight*inputWidth);
    THTensor_(resize4d)(output, T, nOutputPlane, outputHeight, outputWidth);

    THStorage_(clearFlag)(finput_v->storage, TH_STORAGE_REFCOUNTED);
    THStorage_(clearFlag)(finput_h->storage, TH_STORAGE_REFCOUNTED);
    THStorage_(clearFlag)(input->storage, TH_STORAGE_REFCOUNTED);
    THStorage_(clearFlag)(output->storage, TH_STORAGE_REFCOUNTED);
    THStorage_(clearFlag)(tmp_l->storage, TH_STORAGE_REFCOUNTED);
    THStorage_(clearFlag)(tmp_v->storage, TH_STORAGE_REFCOUNTED);
    THStorage_(clearFlag)(weight_l->storage, TH_STORAGE_REFCOUNTED);
    THStorage_(clearFlag)(weight_v->storage, TH_STORAGE_REFCOUNTED);
    THStorage_(clearFlag)(weight_h->storage, TH_STORAGE_REFCOUNTED);

#pragma omp parallel for private(t)
    for(t = 0; t < T; t++)
    {
      THTensor *finput_v_t = THTensor_(newSelect)(finput_v, 0, t);
      THTensor *finput_h_t = THTensor_(newSelect)(finput_h, 0, t);
      THTensor *input_t = THTensor_(newSelect)(input, 0, t);
      THTensor *output_t = THTensor_(newSelect)(output, 0, t);
      THTensor *tmp_l_t = THTensor_(newSelect)(tmp_l, 0, t);
      THTensor *tmp_v_t = THTensor_(newSelect)(tmp_v, 0, t);

      nn_(SpatialFlattenedConvolution_updateOutput_frame)(input_t, output_t,
                                                          tmp_l_t, tmp_v_t,
                                                          finput_v_t, finput_h_t,
                                                          weight_l, weight_v, weight_h,
                                                          bias_l, bias_v, bias_h,
                                                          kW, kH, dW, dH, padding,
                                                          nInputPlane, inputWidth, inputHeight,
                                                          nOutputPlane, outputWidth, outputHeight);

      THTensor_(free)(finput_v_t);
      THTensor_(free)(finput_h_t);
      THTensor_(free)(input_t);
      THTensor_(free)(output_t);
      THTensor_(free)(tmp_l_t);
      THTensor_(free)(tmp_v_t);
    }
    THStorage_(setFlag)(finput_v->storage, TH_STORAGE_REFCOUNTED);
    THStorage_(setFlag)(finput_h->storage, TH_STORAGE_REFCOUNTED);
    THStorage_(setFlag)(input->storage, TH_STORAGE_REFCOUNTED);
    THStorage_(setFlag)(output->storage, TH_STORAGE_REFCOUNTED);
    THStorage_(setFlag)(tmp_l->storage, TH_STORAGE_REFCOUNTED);
    THStorage_(setFlag)(tmp_v->storage, TH_STORAGE_REFCOUNTED);
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
