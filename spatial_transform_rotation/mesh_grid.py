import torch


def create_mesh_grid(size, device):
    """ Creates a mesh grid on device

        Args:
            size(list, torch.Tensor): in the order of [B, C, (D), H, W]

        Return:
            mgrid(torch.Tensor): mesh grid in range 0-max(D,H,W)
    """
    dim = 2 if len(size) == 4 else 3
    B = size[0]
    if dim == 2:
        H, W = size[2], size[3]
        y_pos, x_pos = torch.meshgrid([torch.arange(0, H),
                                       torch.arange(0, W)])
        mgrid = torch.stack([x_pos, y_pos], dim=0)  # [C, H, W]
        mgrid = mgrid.repeat(B,1,1,1)
        mgrid = mgrid.float().to(device)
    else:
        D, H, W = size[2], size[3], size[4]
        z_pos, y_pos, x_pos = torch.meshgrid([torch.arange(0, D),
                                              torch.arange(0, H),
                                              torch.arange(0, W)])
        mgrid = torch.stack([x_pos, y_pos, z_pos], dim=0)  # [C, D, H, W]
        mgrid = mgrid.repeat(B,1,1,1,1)
        mgrid = mgrid.float().to(device)
    return mgrid


def normalize_mesh_grid(mgrid):
    """ Normalizes mesh grid to normalized device coordinates [-1, 1]

        Args:
            mgrid: mesh grid

        Return:
            mgrid_normed(torch.Tensor): mesh grid in range [-1, 1]
    """
    dim = 2 if len(mgrid.size()) == 4 else 3
    if dim == 2:
        H, W = mgrid.size(2), mgrid.size(3)
        mgrid_normed_x = 2.0*mgrid[:, 0:1, ...]/(W-1.0) - 1.0
        mgrid_normed_y = 2.0*mgrid[:, 1:2, ...]/(H-1.0) - 1.0
        mgrid_normed = torch.cat((mgrid_normed_x, mgrid_normed_y), dim=1)
    else:
        D, H, W = mgrid.size(2), mgrid.size(3), mgrid.size(4)
        mgrid_normed_x = 2.0*mgrid[:, 0:1, ...]/(W-1.0) - 1.0
        mgrid_normed_y = 2.0*mgrid[:, 1:2, ...]/(H-1.0) - 1.0
        mgrid_normed_z = 2.0*mgrid[:, 2:3, ...]/(D-1.0) - 1.0
        mgrid_normed = torch.cat((mgrid_normed_x, mgrid_normed_y, mgrid_normed_z),
                                 dim=1)
    return mgrid_normed