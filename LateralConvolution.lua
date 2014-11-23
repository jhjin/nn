local LateralConvolution, parent = torch.class('nn.LateralConvolution', 'nn.SpatialConvolutionLocalMM')

function LateralConvolution:__init(nInputPlane, nOutputPlane, kC)
   parent.__init(self, nInputPlane, nOutputPlane, 1, 1, kC, 1, 1, 0)
end
