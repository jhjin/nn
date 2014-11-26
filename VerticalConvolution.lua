local VerticalConvolution, parent = torch.class('nn.VerticalConvolution', 'nn.SpatialConvolutionLocalMM')

function VerticalConvolution:__init(nInputPlane, nOutputPlane, length, cec)
   assert(nInputPlane == nOutputPlane)
   parent.__init(self, nInputPlane, nOutputPlane, 1, length, 1, 1, 1, 0, cec)
end
