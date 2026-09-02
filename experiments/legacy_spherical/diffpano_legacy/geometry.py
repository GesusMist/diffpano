import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from einops import rearrange


class SphericalFunctions:
    '''all functions are designed for 5D tensor (B, C, F, H, W)'''

    @staticmethod
    def spherical_to_cartesian(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:  # used
        """
        Convert spherical coordinates (theta, phi) to Cartesian coordinates (x, y, z)
        :param theta: Azimuthal angle (N, )
        :param phi: Polar angle (N, )
        :return: Cartesian coordinates (N, 3)
        """
        x = torch.cos(phi) * torch.sin(theta)  # X-axis
        y = torch.sin(phi)                     # Y-axis (Height)
        z = torch.cos(phi) * torch.cos(theta)  # Z-axis
        vec_xyz = torch.stack([x, y, z], dim=-1)
        return vec_xyz

    @staticmethod
    def cartesian_to_spherical(vec_xyz: torch.Tensor) -> List[torch.Tensor]:  # used
        """
        Convert Cartesian coordinates (x, y, z) to spherical coordinates (theta, phi)
        :param vec_xyz: Cartesian coordinates (N, 3)
        :return: theta (N, ), phi (N, )
        """
        x, y, z = vec_xyz[..., 0], vec_xyz[..., 1], vec_xyz[..., 2]

        theta = torch.atan2(x, z)  # theta (Y-axis rotation)
        phi = torch.asin(y)        # phi (X-axis rotation)
        return theta, phi

    @staticmethod
    def rotx(theta: torch.Tensor) -> torch.Tensor:  # used
        """
        Produces a counter-clockwise 3D rotation matrix around the X-axis with angle `theta` in radians.
        :param theta: (B, )
        :return: rotation_matrix: (B, 3, 3)
        """
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        one = torch.ones_like(theta)
        zero = torch.zeros_like(theta)

        row1 = torch.stack([one, zero, zero], dim=-1)   # shape: (..., 3)
        row2 = torch.stack([zero, cos, -sin], dim=-1)
        row3 = torch.stack([zero, sin, cos], dim=-1)

        return torch.stack([row1, row2, row3], dim=-2)

    @staticmethod
    def roty(theta: torch.Tensor) -> torch.Tensor:  # used
        """
        Produces a counter-clockwise 3D rotation matrix around the Y-axis with angle `theta` in radians.
        :param theta: (B, )
        :return: rotation_matrix: (B, 3, 3)
        """
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        one = torch.ones_like(theta)
        zero = torch.zeros_like(theta)

        row1 = torch.stack([cos, zero, -sin], dim=-1)
        row2 = torch.stack([zero, one, zero], dim=-1)
        row3 = torch.stack([sin, zero, cos], dim=-1)

        return torch.stack([row1, row2, row3], dim=-2)

    @staticmethod
    def rotz(theta: torch.Tensor) -> torch.Tensor:  # used
        """
        Produces a counter-clockwise 3D rotation matrix around the Z-axis with angle `theta` in radians.
        :param theta: (B, )
        :return: rotation_matrix: (B, 3, 3)
        """
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        one = torch.ones_like(theta)
        zero = torch.zeros_like(theta)

        row1 = torch.stack([cos, -sin, zero], dim=-1)
        row2 = torch.stack([sin, cos, zero], dim=-1)
        row3 = torch.stack([zero, zero, one], dim=-1)

        return torch.stack([row1, row2, row3], dim=-2)

    @staticmethod
    def rotation_matrix(theta: torch.Tensor, phi: torch.Tensor, roll: float = 0) -> torch.Tensor:  # used
        """
        Returns a camera rotation matrix.

        :param theta: left (negative) to right (positive) [rad], shape = (B, )
        :param phi: upward (negative) to downward (positive) [rad], shape = (B, )
        :param roll: counter-clockwise (negative) to clockwise (positive) [rad], float
        :return: rotation matrix (B, 3, 3)
        """
        roll_tens = torch.tensor(roll, device=theta.device, dtype=theta.dtype)
        roll_tens = roll_tens.repeat(theta.shape[0])

        R = torch.matmul(SphericalFunctions.rotz(
            roll_tens), SphericalFunctions.rotx(phi))
        R = torch.matmul(R, SphericalFunctions.roty(-theta))

        return R

    @staticmethod
    def latlong2world_ours(u, v):  # used
        """Get the (x, y, z, valid) coordinates of the point defined by (u, v)
        for a latlong map."""
        u = u * 2

        # lat-long -> world
        thetaLatLong = torch.pi * (u - 1)
        phiLatLong = torch.pi * v

        x = torch.sin(phiLatLong) * torch.sin(thetaLatLong)
        y = torch.cos(phiLatLong)
        z = -torch.sin(phiLatLong) * torch.cos(thetaLatLong)

        valid = torch.ones(x.shape, dtype=torch.bool)
        return x, y, z, valid

    @staticmethod
    def paste_perspective_to_erp_rectangle(  # used
        panorama: torch.Tensor, square_image: torch.Tensor, view_dir: torch.Tensor, fov: Tuple[float, float] = (80, 80),
        interpolate: bool = True, interpolation_mode: str = 'bilinear',
        add: bool = True, panorama_cnt: Optional[torch.Tensor] = None, return_cnt: bool = False,
        temperature: float = 0.1
    ):
        """
        Paste a batch of square images back onto the panorama at the given view directions
        :param panorama: (b, c, f, h_pan, w_pan)
        :param square_image: (b, c, f, h, w)
        :param view_dir: (b, 3)
        :param fov: Field of view in degrees (float)
        :param add: If True, add the square image to the panorama, else replace
        :param return_indices: If True, return the indices where the square image was pasted
        :param interpolate: If True, uses bilinear interpolation
        :return: Updated panorama (b, c, f, h_pan, w_pan)
        """
        assert interpolate, "Always interpolate for now"
        assert isinstance(fov, (tuple, list)) and len(
            fov) == 2, "fov should be a tuple of two floats"
        assert add, "Always add for now"

        b, c, f, h_pan, w_pan = panorama.shape
        device, dtype = panorama.device, panorama.dtype
        output_size = square_image.shape[-1]
        square_image = square_image.to(device=device, dtype=dtype)

        theta_camera, phi_camera = SphericalFunctions.cartesian_to_spherical(
            view_dir)  # view_dir : (B, 3)
        theta_camera = torch.pi - theta_camera  # ! hard coding
        phi_camera = phi_camera  # ! hard coding
        theta_camera = (theta_camera > torch.pi) * (theta_camera -
                                                    2 * torch.pi) + (theta_camera <= torch.pi) * theta_camera
        phi_camera = (phi_camera > torch.pi / 2) * (phi_camera -
                                                    torch.pi) + (phi_camera <= torch.pi / 2) * phi_camera
        rotation_matrix = SphericalFunctions.rotation_matrix(
            theta_camera, phi_camera)  # (B, 3, 3)

        # camera intrinsics
        # world2image for a camera
        fov_rad = torch.deg2rad(torch.tensor(fov, device=device, dtype=dtype))
        fx = 0.5 / torch.tan(fov_rad[1] / 2)
        fy = 0.5 / torch.tan(fov_rad[0] / 2)
        u0 = 0.5
        v0 = 0.5
        K = torch.tensor([[fx, 0, u0],
                          [0, fy, v0],
                          [0, 0, 1]])
        K = K.to(device=device, dtype=dtype).repeat(b, 1, 1)  # (B, 3, 3)
        # M = torch.einsum('b i j, b j k-> b i k', K, rotation_matrix.permute(0, 2, 1))  # (B, 3, 3)
        M = torch.einsum('b i j, b j k-> b i k', K,
                         rotation_matrix)  # (B, 3, 3)

        # get pixel_dirs from panorama
        u_range = torch.linspace(
            0, 1, w_pan * 2 + 1, device=device, dtype=dtype)
        v_range = torch.linspace(
            0, 1, h_pan * 2 + 1, device=device, dtype=dtype)
        u_range = u_range[1::2]
        v_range = v_range[1::2]
        u, v = torch.meshgrid(u_range, v_range, indexing='xy')
        dx, dy, dz, valid = SphericalFunctions.latlong2world_ours(u, v)
        xyz = torch.stack([dx, dy, dz], dim=0)
        xyz = rearrange(xyz, 'i h w -> 1 i (h w)').repeat(b,
                                                          1, 1)  # (B, 3, h*w)

        forward_vector = torch.tensor(
            [0, 0, -1], device=device, dtype=dtype).unsqueeze(0)
        forward_vector = forward_vector.repeat(b, 1)
        forward_vector = torch.einsum('b i j, b j -> b i', rotation_matrix.permute((0, 2, 1)), forward_vector)
        mask = torch.einsum('b i N, b i -> b N', xyz, forward_vector) > 0

        panorama_cnt = torch.zeros_like(
            panorama) if panorama_cnt is None else panorama_cnt
        panorama = rearrange(panorama, 'b c f h w -> b c f (h w)')
        panorama_cnt = rearrange(panorama_cnt, 'b c f h w -> b c f (h w)')
        for idx_b in range(b):
            xyz_selected = xyz[idx_b, :, mask[idx_b]]
            # xyz_selected = xyz[idx_b]  # no masking
            perspective_xyz = torch.einsum('i j, j N -> i N', M[idx_b], xyz_selected)
            perspective_u = perspective_xyz[0] / perspective_xyz[2]
            perspective_v = perspective_xyz[1] / perspective_xyz[2]
            # print(perspective_u.min(), perspective_u.max())
            perspective_uv = torch.stack(
                [perspective_u, perspective_v], dim=-1)
            perspective_uv = perspective_uv.unsqueeze(0).unsqueeze(0)
            perspective_uv = perspective_uv * 2 - 1

            us = perspective_uv[..., 0]
            vs = perspective_uv[..., 1]
            us = - us

            perspective_uv = torch.stack([us, vs], dim=-1)

            _square_image = rearrange(
                square_image[idx_b], 'c f h w -> f c h w')

            output = torch.cat([F.grid_sample(
                _square_image[idx_f][None],
                perspective_uv,
                padding_mode='border',
                mode=interpolation_mode,  # or nearest
                align_corners=True
            ) for idx_f in range(f)], dim=0)

            channel_square, height_square, width_square = _square_image.shape[1:]
            grid_h = torch.linspace(-1, 1, height_square,
                                    device=device, dtype=dtype)
            grid_w = torch.linspace(-1, 1, width_square,
                                    device=device, dtype=dtype)
            grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing='ij')
            cur_coord_pers = torch.stack([grid_w, grid_h], dim=-1)
            cur_coord_pers = rearrange(cur_coord_pers, 'h w d -> 1 1 h w d')
            weight = torch.exp(-torch.norm(cur_coord_pers,dim=-1) / temperature)
            output_cnt = F.grid_sample(
                weight[0][None],
                perspective_uv,
                padding_mode='zeros',
                mode='bilinear',
                align_corners=True
            )  # (1, 1, H, W)

            output = rearrange(output * output_cnt, 'f c h w -> c f h w')
            output_cnt = rearrange(output_cnt, 'f c h w -> c f h w')
            if add:
                panorama[idx_b, ..., mask[idx_b]] += output.squeeze(2)
                panorama_cnt[idx_b, ..., mask[idx_b]] += output_cnt.squeeze(2)
            else:
                panorama[idx_b, ..., mask[idx_b]] = (output.squeeze(2)).to(dtype=panorama.dtype)
                panorama_cnt[idx_b, ..., mask[idx_b]] = (output_cnt.squeeze(2)).to(dtype=panorama_cnt.dtype)

        panorama = rearrange(panorama, 'b c f (h w) -> b c f h w', h=h_pan)
        panorama_cnt = rearrange(panorama_cnt, 'b c f (h w) -> b c f h w', h=h_pan)

        if return_cnt:
            return panorama, panorama_cnt
        else:
            return panorama

    @staticmethod
    def fibonacci_sphere(N: int, randomize: bool = False) -> torch.Tensor:  # used
        """
        Generate N uniformly distributed points on a unit sphere using PyTorch.
        :param N: Number of points (int)
        :param randomize: If True, introduces a random offset
        :return: directions (N, 3)
        """
        phi = (1 + torch.sqrt(torch.tensor(5.0))) / 2  # Golden ratio
        indices = torch.arange(0, N, dtype=torch.float32)

        if randomize:
            # Random offset for better uniformity
            offset = torch.rand(1) * 2 - 1 / N
        else:
            offset = 0

        theta = 2 * torch.pi * indices / phi  # Azimuthal angle
        z = 1 - (2 * (indices + offset) / N)  # z-coordinate
        radius = torch.sqrt(1 - z ** 2)  # Radius in xy-plane

        x = radius * torch.cos(theta)
        y = radius * torch.sin(theta)

        directions = torch.stack([x, y, z], dim=1)
        return directions

    @staticmethod
    def horizontal_and_vertical_view_dirs_v3_fov_xy_dense_equator(  # used
        fov_single: float = 80, fov_overlap_x: float = 0.6, fov_overlap_y: float = 0.6, n_theta_offset: int = 3,
    ):
        """ Generate horizontal view directions """
        fov_overlap_x = fov_single * fov_overlap_x
        fov_overlap_y = fov_single * fov_overlap_y

        phis = torch.linspace(0, 90, math.ceil((90 + fov_single / 2) / (fov_single - fov_overlap_y)))
        phis = [phi.item() for phi in phis]
        phis = phis + [-phi for phi in phis[1:]]
        phis.sort(key=lambda x: abs(x + 1e-2))
        phis = [float(phi) for phi in phis]

        view_dirs = []

        for phi in phis:
            total_len = math.cos(math.radians(phi)) * 360
            n_theta = total_len / (fov_single - fov_overlap_x)
            n_theta = math.ceil(n_theta) + n_theta_offset

            theta = torch.linspace(-math.pi, math.pi, n_theta + 1)[:-1]
            phi = torch.ones_like(theta) * (phi / 180 * math.pi)
            view_dirs.append(SphericalFunctions.spherical_to_cartesian(theta, phi))

        view_dirs = torch.cat(view_dirs, dim=0)
        return view_dirs

    @staticmethod
    def extract_perspective_from_spherical_rectangle_rasterize(  # used
        spherical_points: torch.Tensor, view_dir: torch.Tensor,
        fov: tuple = (80, 80), output_size: tuple = (512, 512),
        device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """
        Extract a square image using colors from the spherical representation based on the given batch view direction
        spherical_points: (B, F, N, 3)
        view_dir: (B, 3)
        """
        # B = view_dir.shape[0]
        assert isinstance(fov, (tuple, list)) and len(fov) == 2, "Field of view must be a tuple of two values"
        assert isinstance(output_size, (tuple, list)) and len(output_size) == 2, "Output size must be a tuple of two values"

        fov_rad = torch.deg2rad(torch.tensor(fov, device=device, dtype=dtype))
        x_range = torch.linspace(torch.tan(fov_rad[1] / 2), -torch.tan(fov_rad[1] / 2), output_size[1], device=device, dtype=dtype)
        y_range = torch.linspace(torch.tan(fov_rad[0] / 2), -torch.tan(fov_rad[0] / 2), output_size[0], device=device, dtype=dtype)

        xv, yv = torch.meshgrid(x_range, y_range, indexing='xy')
        zv = torch.ones_like(xv)
        pixel_dirs = torch.stack([xv, yv, -zv], dim=-1)
        pixel_dirs = pixel_dirs / torch.linalg.norm(pixel_dirs, dim=-1, keepdim=True)

        theta, phi = SphericalFunctions.cartesian_to_spherical(view_dir)  # view_dir : (B, 3)
        rotation_matrix = SphericalFunctions.rotation_matrix(theta, phi)  # (B, 3, 3)
        rotated_dirs = torch.einsum('bij,hwi->bhwj', rotation_matrix, pixel_dirs)

        tmp_spherical_points = spherical_points[:, 0, :, :].clone()
        indices = []
        for idx_rot_i in range(rotated_dirs.shape[1]):
            indices_h = []
            for idx_rot_j in range(rotated_dirs.shape[2]):
                rotated_dir = rotated_dirs[:, idx_rot_i, idx_rot_j, :]  # (B, 3)
                index = torch.argmin(torch.linalg.norm(rotated_dir.unsqueeze(1) - tmp_spherical_points, dim=-1), dim=1)  # (B, )
                indices_h.append(index)
                tmp_spherical_points[:, index, :] = - rotated_dir
            indices.append(torch.stack(indices_h, dim=-1))
        indices = torch.stack(indices, dim=1)
        indices = rearrange(indices, 'b h w -> b (h w)')[0]
        indices = indices.to(dtype=torch.long, device=device)

        return indices

    @staticmethod
    def world_to_perspective(  # used
        xyz: torch.Tensor, view_dir: torch.Tensor, fov: Tuple[float, float]
    ) -> torch.Tensor:
        '''
        xyz: (B, N, 3)
        '''
        device, dtype = xyz.device, xyz.dtype
        batch_size = xyz.size(0)
        assert batch_size == 1, "Batch size must be 1"

        theta_camera, phi_camera = SphericalFunctions.cartesian_to_spherical(view_dir)  # view_dir : (B, 3)
        theta_camera = (theta_camera > torch.pi) * (theta_camera - 2 * torch.pi) + (theta_camera <= torch.pi) * theta_camera
        phi_camera = (phi_camera > torch.pi / 2) * (phi_camera - torch.pi) + (phi_camera <= torch.pi / 2) * phi_camera
        rotation_matrix = SphericalFunctions.rotation_matrix(theta_camera, phi_camera)  # (B, 3, 3)

        # camera intrinsics
        # world2image for a camera
        fov_rad = torch.deg2rad(torch.tensor(fov, device=device, dtype=dtype))
        fx = 0.5 / torch.tan(fov_rad[1] / 2)
        fy = 0.5 / torch.tan(fov_rad[0] / 2)
        u0 = 0.
        v0 = 0.
        K = torch.tensor([[fx, 0, u0],
                          [0, fy, v0],
                          [0, 0, 1]])
        K = K.to(device=device, dtype=dtype).repeat(batch_size, 1, 1)  # (B, 3, 3)
        M = torch.einsum('b i j, b j k-> b i k', K, rotation_matrix)  # (B, 3, 3)

        xyz = rearrange(xyz, 'b n i -> b i n')

        forward_vector = torch.tensor([0, 0, -1], device=device, dtype=dtype).unsqueeze(0)
        forward_vector = forward_vector.repeat(batch_size, 1)
        forward_vector = torch.einsum('b i j, b j -> b i', rotation_matrix.permute((0, 2, 1)), forward_vector)  # ! hard coding
        mask_hemisphere = torch.einsum('b i N, b i -> b N', xyz, forward_vector) > 0

        coord_pers = []
        for idx_b in range(batch_size):
            perspective_xyz = torch.einsum('i j, j N -> i N', M[idx_b], xyz[idx_b])
            eps = 1e-6
            perspective_u = perspective_xyz[0] / (perspective_xyz[2] + eps)
            perspective_v = perspective_xyz[1] / (perspective_xyz[2] + eps)
            perspective_uv = torch.stack([perspective_u, perspective_v], dim=-1)
            perspective_uv = perspective_uv.unsqueeze(0).unsqueeze(0)
            perspective_uv = perspective_uv * 2

            us = perspective_uv[..., 0]
            vs = perspective_uv[..., 1]
            us = - us  # ! hard coding

            perspective_uv = torch.stack([us, vs], dim=-1)  # (1, 1, N, 2)

            mask = mask_hemisphere[idx_b][None, None, :]
            perspective_uv = perspective_uv * \
                mask[..., None] + \
                torch.ones_like(perspective_uv) * (-100) * (~mask[..., None])
            coord_pers.append(perspective_uv)

        coord_pers = torch.stack(coord_pers, dim=0)  # (B, 1, 1, N, 2)
        coord_pers = coord_pers.squeeze(1, 2)  # (B, N, 2)
        coord_pers = coord_pers.squeeze(0)  # (N, 2)

        return coord_pers

    @staticmethod
    def discretize_spherical_points(  # used
        coord_pers: torch.Tensor,
        height: int, width: int,
    ) -> torch.Tensor:
        '''
        spherical_points: (N, 2)
        height: int
        width: int

        return indices_reorder
        '''
        assert height == width, "Height and width must be the same for now"
        is_even = height % 2 == 0
        # assert is_even, "Height must be even for now"
        if is_even:
            num_nns = [(idx * 2) ** 2 - ((idx - 1) * 2) **
                       2 for idx in range(1, height // 2 + 1)]
        else:
            num_nns = [1] + [(idx * 2 + 1) ** 2 - ((idx - 1) * 2 + 1)
                             ** 2 for idx in range(1, height // 2 + 1)]
        assert sum(num_nns) == height * \
            width, "Sum of the number of nearest neighbors must be equal to the total number of points"

        indices = torch.arange(coord_pers.size(0), device=coord_pers.device)
        distance_from_origin = coord_pers.norm(dim=-1)
        indices_from_origin = torch.argsort(
            distance_from_origin, descending=False)

        indices_reorder = torch.ones(
            (height * width, ), device=coord_pers.device, dtype=torch.long) * -1
        for num_nn in num_nns:
            cur_indices = indices_from_origin[:num_nn]
            cur_coord_pers = coord_pers[cur_indices]
            cur_coord_pers = (cur_coord_pers - cur_coord_pers.min()) / \
                (cur_coord_pers.max() - cur_coord_pers.min())
            cur_coord_pers = cur_coord_pers * 2 - 1

            if is_even:
                target_coord_pers = torch.cat([
                    torch.stack([torch.ones(num_nn // 4, device=coord_pers.device) * -1, torch.linspace(-1, 1, num_nn // 4 + 1, device=coord_pers.device)[:-1]], dim=1),
                    torch.stack([torch.linspace(-1, 1, num_nn // 4 + 1, device=coord_pers.device)[:-1], torch.ones(num_nn // 4, device=coord_pers.device) * 1], dim=1),
                    torch.stack([torch.ones(num_nn // 4, device=coord_pers.device) * 1, torch.linspace(1, -1, num_nn // 4 + 1, device=coord_pers.device)[:-1]], dim=1),
                    torch.stack([torch.linspace(1, -1, num_nn // 4 + 1, device=coord_pers.device)[:-1], torch.ones(num_nn // 4, device=coord_pers.device) * -1], dim=1),
                ])

                idx_h_min, idx_h_max = height // 2 - (num_nn // 4) // 2 - 1, height // 2 + (num_nn // 4) // 2
                idx_w_min, idx_w_max = width // 2 - (num_nn // 4) // 2 - 1, width // 2 + (num_nn // 4) // 2
                target_indices = torch.cat([
                    torch.arange(idx_w_min, idx_w_max, device=coord_pers.device) + idx_h_min * width,
                    torch.arange(idx_h_min, idx_h_max, device=coord_pers.device) * width + idx_w_max,
                    torch.arange(idx_w_max, idx_w_min, step=-1, device=coord_pers.device) + idx_h_max * width,
                    torch.arange(idx_h_max, idx_h_min, step=-1, device=coord_pers.device) * width + idx_w_min,
                ])
            else:
                if num_nn == 1:
                    target_coord_pers = torch.tensor([[0, 0]], device=coord_pers.device)
                    target_indices = torch.tensor([height // 2 * width + width // 2], device=coord_pers.device)
                else:
                    target_coord_pers = torch.cat([
                        torch.stack([torch.ones(num_nn // 4, device=coord_pers.device) * -1, torch.linspace(-1, 1, num_nn // 4 + 1, device=coord_pers.device)[:-1]], dim=1),
                        torch.stack([torch.linspace(-1, 1, num_nn // 4 + 1, device=coord_pers.device)[:-1], torch.ones(num_nn // 4, device=coord_pers.device) * 1], dim=1),
                        torch.stack([torch.ones(num_nn // 4, device=coord_pers.device) * 1, torch.linspace(1, -1, num_nn // 4 + 1, device=coord_pers.device)[:-1]], dim=1),
                        torch.stack([torch.linspace(1, -1, num_nn // 4 + 1, device=coord_pers.device)[:-1], torch.ones(num_nn // 4, device=coord_pers.device) * -1], dim=1),
                    ])

                    idx_h_min, idx_h_max = height // 2 - (num_nn // 4) // 2, height // 2 + (num_nn // 4) // 2
                    idx_w_min, idx_w_max = width // 2 - (num_nn // 4) // 2, width // 2 + (num_nn // 4) // 2
                    target_indices = torch.cat([
                        torch.arange(idx_w_min, idx_w_max, device=coord_pers.device) + idx_h_min * width,
                        torch.arange(idx_h_min, idx_h_max, device=coord_pers.device) * width + idx_w_max,
                        torch.arange(idx_w_max, idx_w_min, step=-1, device=coord_pers.device) + idx_h_max * width,
                        torch.arange(idx_h_max, idx_h_min, step=-1, device=coord_pers.device) * width + idx_w_min,
                    ])

            # cur_indices to target_indices
            for target_idx, target_coord in zip(target_indices, target_coord_pers):
                selected_idx = torch.argmin((cur_coord_pers - target_coord).norm(dim=-1))
                indices_reorder[target_idx] = cur_indices[selected_idx]
                cur_coord_pers[selected_idx] = torch.ones(2, device=coord_pers.device) * 1000

            indices_from_origin = indices_from_origin[num_nn:]

        assert (indices_reorder == -1).sum() == 0, "All indices must be filled"

        return indices_reorder

    @staticmethod
    def get_prompt_indices(view_dir, prompt_dir, prompt_fovs):  # used
        '''
        view_dir: (N, 3)
        prompt_dir: (K, 3)
        prompt_fovs: (K, 2)
        '''
        # find the closest prompt direction
        cosine_sim = torch.einsum('n i, k i -> n k', view_dir, prompt_dir)
        _, indices = cosine_sim.topk(1, dim=-1)  # (N, 1)
        indices = indices.squeeze(-1)
        fovs = [prompt_fovs[idx] for idx in indices]
        return indices, fovs

    @staticmethod
    def get_height_width_from_fov(fov, num_points_on_sphere=2600):  # used
        aspect_ratio = fov[1] / fov[0]
        A_FOV_ratio = 4 * math.sin(fov[1] / 180 * math.pi / 2) * math.sin(fov[0] / 180 * math.pi / 2) / (4 * math.pi)
        latent_height = round(math.sqrt(num_points_on_sphere * A_FOV_ratio / aspect_ratio))
        latent_width = round(aspect_ratio * latent_height)
        return latent_height, latent_width

    @staticmethod
    def dynamic_latent_sampling(  # used
        spherical_points: torch.Tensor,
        cur_view_dir: torch.Tensor,
        num_points_on_sphere: int,
        _fov: Tuple[float, float],
        temperature: float,
        center_first: bool = True,
    ):
        device, dtype = spherical_points.device, spherical_points.dtype

        xyz_coord_pers = SphericalFunctions.world_to_perspective(
            spherical_points[:, 0, :, :].to(dtype=torch.float), cur_view_dir.to(dtype=torch.float), fov=_fov
        )  # (N, 2)
        assert not torch.isnan(xyz_coord_pers).any(), 'xyz_coord_pers contains NaN values'

        # dynamic H, W
        latent_height, latent_width = SphericalFunctions.get_height_width_from_fov(_fov, num_points_on_sphere)
        indices_in_square = SphericalFunctions.extract_perspective_from_spherical_rectangle_rasterize(
            spherical_points, cur_view_dir, fov=_fov,
            output_size=(latent_height, latent_width),
            device=device, dtype=dtype
        )  # (M, )

        if center_first:
            selected_coords = xyz_coord_pers[indices_in_square]
            x_min, y_min = selected_coords.min(dim=0)[0]  # (2,)
            x_max, y_max = selected_coords.max(dim=0)[0]  # (2,)
            in_bbox = ((xyz_coord_pers[:, 0] >= x_min) & (xyz_coord_pers[:, 0] <= x_max) &
                       (xyz_coord_pers[:, 1] >= y_min) & (xyz_coord_pers[:, 1] <= y_max))
            indices_new = torch.where(in_bbox)[0]

            n_points = len(indices_new)
            aspect_ratio = _fov[1] / _fov[0]
            cur_latent_height = math.floor(math.sqrt(n_points / aspect_ratio))
            cur_latent_width = math.floor(math.sqrt(n_points * aspect_ratio))

            cur_coord_pers = xyz_coord_pers[indices_new]
            cur_coord_pers[..., 0] = - cur_coord_pers[..., 0]  # ! needed
            cur_coord_pers = torch.stack([cur_coord_pers[:, 1], cur_coord_pers[:, 0]], dim=-1)  # (N, 2)
            indices_reorder = SphericalFunctions.discretize_spherical_points(cur_coord_pers, cur_latent_height, cur_latent_width)
            indices_new = indices_new[indices_reorder]
        else:
            indices_new = indices_in_square

        cur_coord_pers = xyz_coord_pers[indices_new]
        weight = torch.exp(-torch.norm(cur_coord_pers, dim=-1) / temperature)  # (N, )
        weight = weight[None, None, :]  # (1, 1, N)

        return indices_new, weight


# Backward-compatible alias for the original public typo.
SphericalFunctions.dynamic_laetent_sampling = SphericalFunctions.dynamic_latent_sampling
