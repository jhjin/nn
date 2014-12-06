local SpatialFlattenedConvolution, parent = torch.class('nn.SpatialFlattenedConvolution', 'nn.Module')

function SpatialFlattenedConvolution:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padding)
   parent.__init(self)

   dW = dW or 1
   dH = dH or 1

   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.kW = kW
   self.kH = kH

   self.dW = dW
   self.dH = dH
   self.padding = padding or 0

   self.weight_l = torch.Tensor(nOutputPlane, nInputPlane)
   self.weight_v = torch.Tensor(nOutputPlane, kH)
   self.weight_h = torch.Tensor(nOutputPlane, kW)

   self.bias_l = torch.Tensor(nOutputPlane)
   self.bias_v = torch.Tensor(nOutputPlane)
   self.bias_h = torch.Tensor(nOutputPlane)

   self.tmp_l = torch.Tensor()
   self.tmp_v = torch.Tensor()

   self.finput_v = torch.Tensor()
   self.finput_h = torch.Tensor()

   self:reset()
end

function SpatialFlattenedConvolution:reset(stdv, mode)
   local mode = mode or 'heuristic'

   if stdv then
      stdv_l = stdv * math.sqrt(3)
      stdv_v = stdv * math.sqrt(3)
      stdv_h = stdv * math.sqrt(3)
   elseif mode == 'heuristic' then
      stdv_l = math.sqrt(1/self.nInputPlane)
      stdv_v = math.sqrt(1/self.kH)
      stdv_h = math.sqrt(1/self.kW)
   elseif mode == 'xavier' then
      stdv_l = math.sqrt(6/(self.nInputPlane + self.nOutputPlane))
      stdv_v = math.sqrt(6/(self.kH + self.kH))
      stdv_h = math.sqrt(6/(self.kW + self.kW))
   else
      assert(false)
   end

   self.weight_l:uniform(-stdv_l, stdv_l)
   self.weight_v:uniform(-stdv_v, stdv_v)
   self.weight_h:uniform(-stdv_h, stdv_h)

   self.bias_l:uniform(-stdv_l, stdv_l)
   self.bias_v:uniform(-stdv_v, stdv_v)
   self.bias_h:uniform(-stdv_h, stdv_h)
end

function SpatialFlattenedConvolution:updateOutput(input)
   return input.nn.SpatialFlattenedConvolution_updateOutput(self, input)
end

function SpatialFlattenedConvolution:updateGradInput(input, gradOutput)
   return assert(false)
end

function SpatialFlattenedConvolution:accGradParameters(input, gradOutput, scale)
   return assert(false)
end
