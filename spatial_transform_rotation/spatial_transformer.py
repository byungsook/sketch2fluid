import torch
import torch.nn.functional as F
from spatial_transform_rotation.mesh_grid import create_mesh_grid, normalize_mesh_grid
from spatial_transform_rotation.padding import pad_zeros
from spatial_transform_rotation.rotation_matrix import rotation_matrix, scale_matrix

class SpatialTransformer(torch.nn.Module):
    """
    Creates grid necessary for matrix transformation which can be reused multiple times
    for that transformation into specified tensor shape. Compared to directly using apply_matrix_transform
    this saves memory if the transformation is going to be applied to tensors of always same size

    # TODO:
        duplicate batch size and apply multiple transformation at same time?
        More general spatial transformation than rotation
    """
    def __init__(self, device, tensor_shape=[1, 1, 129, 129, 129]):
        super().__init__()
        
        self.device = device
        self.tensor_shape = tensor_shape
        self.positions = create_mesh_grid(self.tensor_shape, self.device)
        self.ndc = normalize_mesh_grid(self.positions)
        self.ndc = self.ndc.permute(0, 2, 3, 4, 1)  # -> [B,D,W,H,C]

    def forward(self, tensor, roll=.0, pitch=.0, yaw=.0, margin_horizontal_plane=0.0, order='rpy'): # order-dependent for ratate back
        r = rotation_matrix(roll, pitch, yaw, order)
        r = torch.tensor(r, requires_grad=False, device=tensor.device).float()
        t2 = self.apply_matrix_transform(tensor, r, margin_horizontal_plane=margin_horizontal_plane)
        return t2

    def apply_matrix_transform(self, tensor, matrix, margin_horizontal_plane=0.0):
        '''
            :param tensor: with shape [B,C,D,H,W]
            :param matrix: 3x3 spatial transformation matrix (eg. rotation matrix)
            :return: tensor after its transformed with the matrix
            :param margin_horizontal_plane: can be used to avoid cropping
        '''

        # grid_shape = list(tensor.shape)
        # if margin_horizontal_plane > 0.0:
        #     grid_shape[2] = int(math.ceil(grid_shape[2] * margin_horizontal_plane))
        #     grid_shape[4] = int(math.ceil(grid_shape[4] * margin_horizontal_plane))

        # tensor_pad = pad_zeros(target_shape=grid_shape, x=tensor)
        # transformed_coordinates = self.transformation_grid(grid_shape, tensor.device, matrix)
        
        tensor_pad = pad_zeros(target_shape=self.tensor_shape, x=tensor)
        transformed_coordinates = self.transformation_grid(self.tensor_shape, tensor.device, matrix)
        
        transformed_coordinates = transformed_coordinates.repeat(tensor_pad.size(0),1,1,1,1)
        # print (transformed_coordinates.shape)
        transformed_values = F.grid_sample(tensor_pad, transformed_coordinates)  # grid=[N,C,X,Y,Z]
        return transformed_values

    def transformation_grid(self, tensor_shape, device, matrix):
        # positions = create_mesh_grid(tensor_shape, device)
        # ndc = normalize_mesh_grid(positions)
        # ndc = ndc.permute(0, 2, 3, 4, 1)  # -> [B,D,W,H,C]
        transformed_coordinates = torch.matmul(self.ndc.to(device), matrix)
        return transformed_coordinates





