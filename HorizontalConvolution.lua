local HorizontalConvolution, parent = torch.class('nn.HorizontalConvolution', 'nn.SpatialConvolutionOneToOneMM')

function HorizontalConvolution:__init(nInputPlane, nOutputPlane, length)
   parent.__init(self, nInputPlane, nOutputPlane, length, 1, 1, 1, 0)
end
