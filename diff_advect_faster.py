import torch
import torch.nn.functional as F

class DiffAdvectShapeless:
    def __init__(self, device):
        self.device = device
        self.advector_dict = {}

    def _get_advect_module(self, shape):
        if shape in self.advector_dict:
            return self.advector_dict[shape]

        else:
            a = DiffAdvectFaster(self.device, size=[1, 1, *shape])
            self.advector_dict[shape] = a
            return a

    def __call__(self, vel, den, dt, dx=1):
        return self._get_advect_module(vel.shape[2:])(vel, den, dt, dx)

    def concat_velocities(self, vel_list, dt):
        return self._get_advect_module(vel_list[0].shape[2:]).concat_velocities(vel_list, dt)

    def mgrid(self, s):
        return self._get_advect_module((s,s,s)).mgrid

class DiffAdvectFaster(torch.nn.Module):
    
    def __init__(self, device, size=[1, 1, 129, 129, 129]):
        super().__init__()
        
        self.device = device
        self.size = size
        self.mgrid = self.create_mesh_grid(self.size)

    def forward(self, vel, den, dt, dx=1):
        return self.grid_sample(den, self.mgrid.to(vel.device) - vel*dt, dx)

    def create_mesh_grid(self, size):

        B = size[0]
        D, H, W = size[2], size[3], size[4]
        
        z_pos, y_pos, x_pos = torch.meshgrid([torch.arange(0, D),
                                              torch.arange(0, H), 
                                              torch.arange(0, W)])
        mgrid = torch.stack([x_pos, y_pos, z_pos], dim=0)  # [C, D, H, W]
        mgrid = mgrid.unsqueeze(0).repeat(B,1,1,1,1)    # added: [B, C, D, H, W]
        mgrid = mgrid.float().to(self.device)
        return mgrid

    def normalize_mesh_grid(self, mgrid):
        
        D, H, W = mgrid.size(2), mgrid.size(3), mgrid.size(4)
        # print (D,H,W)
        mgrid_normed_x = 2.0*mgrid[:,0:1, ...]/(W-1.0) - 1.0
        mgrid_normed_y = 2.0*mgrid[:,1:2, ...]/(H-1.0) - 1.0
        mgrid_normed_z = 2.0*mgrid[:,2:3, ...]/(D-1.0) - 1.0
        mgrid_normed = torch.cat((mgrid_normed_x, mgrid_normed_y, mgrid_normed_z), dim=1)   # dim=0
        return mgrid_normed
    
    def grid_sample(self, grid, back_trace, dx):
        dim = 2 if len(grid.size()) == 3 else 3
        back_trace_normed = self.normalize_mesh_grid(back_trace)
        permutation = (0, 2, 3, 4, 1)  # new
        back_trace_normed = back_trace_normed.permute(permutation) / dx  # [N, (D), H, W, C]
        grid_sampled = F.grid_sample(grid, back_trace_normed, mode='bilinear', padding_mode='zeros')
        return grid_sampled


    def concat_velocities(self, vel_list, dt):
        """
        Concatenate a list of velocities in a single one.
        Not implemented for patch advection

        dim: (bs, 3, x, y, z)
        """

        # sum_lagrange (i in [0, n[) v_i = sum (i in [0, n[) v_i[x-dt*v i->n] 

        v = vel_list[-1]
        
        for i in range(len(vel_list)-2, 0, -1):
            v = v + self(v, vel_list[i], dt)
        
        return v


    # def advect_semi_lagrange(vel, rho, args, order=1, clamp_mode=2, rk_order=3):
    # def advect_semi_lagrange(vel, rho, dt, dx = 1):

    #     mgrid = create_mesh_grid(rho.size())
    #     # print (mgrid.size())
    #     return grid_sample(rho, mgrid - vel*dt, dx)

