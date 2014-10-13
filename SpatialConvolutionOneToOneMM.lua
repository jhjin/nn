local SpatialConvolutionOneToOneMM, parent = torch.class('nn.SpatialConvolutionOneToOneMM', 'nn.Module')

function SpatialConvolutionOneToOneMM:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padding)
   parent.__init(self)

   dW = dW or 1
   dH = dH or 1

   assert(nInputPlane == nOutputPlane)
   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.kW = kW
   self.kH = kH

   self.dW = dW
   self.dH = dH
   self.padding = padding or 0

   self.weight = torch.Tensor(nOutputPlane, kH*kW)
   self.bias = torch.Tensor(nOutputPlane)
   self.gradWeight = torch.Tensor(nOutputPlane, kH*kW)
   self.gradBias = torch.Tensor(nOutputPlane)

   self.finput = torch.Tensor()
   self.fgradInput = torch.Tensor()

   self:reset()
end

function SpatialConvolutionOneToOneMM:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(self.kW*self.kH*self.nInputPlane)
   end
   if nn.oldSeed then
      self.weight:apply(function()
         return torch.uniform(-stdv, stdv)
      end)
      self.bias:apply(function()
         return torch.uniform(-stdv, stdv)
      end)
   else
      self.weight:uniform(-stdv, stdv)
      self.bias:uniform(-stdv, stdv)
   end
end

function SpatialConvolutionOneToOneMM:updateOutput(input)
   return input.nn.SpatialConvolutionOneToOneMM_updateOutput(self, input)
end

function SpatialConvolutionOneToOneMM:updateGradInput(input, gradOutput)
   if self.gradInput then
      return input.nn.SpatialConvolutionOneToOneMM_updateGradInput(self, input, gradOutput)
   end
end

function SpatialConvolutionOneToOneMM:accGradParameters(input, gradOutput, scale)
   return input.nn.SpatialConvolutionOneToOneMM_accGradParameters(self, input, gradOutput, scale)
end
