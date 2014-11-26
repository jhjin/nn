local HorizontalConvolution, parent = torch.class('nn.HorizontalConvolution', 'nn.SpatialConvolutionLocalMM')

function HorizontalConvolution:__init(nInputPlane, nOutputPlane, length, cec)
   assert(nInputPlane == nOutputPlane)
   parent.__init(self, nInputPlane, nOutputPlane, length, 1, 1, 1, 1, 0, cec)
end
