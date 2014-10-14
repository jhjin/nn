local VerticalConvolution, parent = torch.class('nn.VerticalConvolution', 'nn.SpatialConvolutionOneToOneMM')

function VerticalConvolution:__init(nInputPlane, nOutputPlane, length)
   parent.__init(self, nInputPlane, nOutputPlane, 1, length, 1, 1, 0)
end
