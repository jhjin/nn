local LateralConvolution, parent = torch.class('nn.LateralConvolution', 'nn.SpatialConvolutionLocalMM')

function LateralConvolution:__init(nInputPlane, nOutputPlane, length, cec)
   assert(length <= nInputPlane)
   parent.__init(self, nInputPlane, nOutputPlane, 1, 1, length, 1, 1, 0, cec)
end
