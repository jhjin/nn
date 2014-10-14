local LateralConvolution, parent = torch.class('nn.LateralConvolution', 'nn.SpatialConvolutionMM')

function LateralConvolution:__init(nInputPlane, nOutputPlane)
   parent.__init(self, nInputPlane, nOutputPlane, 1, 1, 1, 1, 0)
end
