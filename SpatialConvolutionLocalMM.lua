local SpatialConvolutionLocalMM, parent = torch.class('nn.SpatialConvolutionLocalMM', 'nn.Module')

function SpatialConvolutionLocalMM:__init(nInputPlane, nOutputPlane, kW, kH, kC, dW, dH, padding)
   parent.__init(self)

   dW = dW or 1
   dH = dH or 1

   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.kW = kW
   self.kH = kH
   self.kC = kC

   self.dW = dW
   self.dH = dH
   self.padding = padding or 0

   self.weight = torch.Tensor(nOutputPlane, nInputPlane*kH*kW)
   self.bias = torch.Tensor(nOutputPlane)
   self.gradWeight = torch.Tensor(nOutputPlane, nInputPlane*kH*kW)
   self.gradBias = torch.Tensor(nOutputPlane)

   self.finput = torch.Tensor()
   self.fgradInput = torch.Tensor()

   self.w_indicator = torch.range(0,nOutputPlane-1):mul(nInputPlane-kC):div(math.max(1,nOutputPlane-1)):round()

   self:reset()
end

function SpatialConvolutionLocalMM:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(self.kW*self.kH*self.kC)
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

function SpatialConvolutionLocalMM:updateOutput(input)
   return input.nn.SpatialConvolutionLocalMM_updateOutput(self, input)
end

function SpatialConvolutionLocalMM:updateGradInput(input, gradOutput)
   if self.gradInput then
      return input.nn.SpatialConvolutionLocalMM_updateGradInput(self, input, gradOutput)
   end
end

function SpatialConvolutionLocalMM:accGradParameters(input, gradOutput, scale)
   return input.nn.SpatialConvolutionLocalMM_accGradParameters(self, input, gradOutput, scale)
end
