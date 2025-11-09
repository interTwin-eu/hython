"""Process-based Models"""


#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Hylke E. Beck"
__email__ = "hylke.beck@gmail.com"
__date__ = "May 20, 2019"

import numpy as np
from typing import Any, Optional, Union

import torch

import torch
import torch.nn.functional as F


def uh_gamma(a, b, lenF=10):
    """Unit hydrograph [time (same all time steps), batch, var]."""
    m = a.shape
    lenF = min(a.shape[0], lenF)
    w = torch.zeros([lenF, m[1], m[2]])
    aa = (
        F.relu(a[0:lenF, :, :]).view([lenF, m[1], m[2]]) + 0.1
    )  # minimum 0.1. First dimension of a is repeat
    theta = F.relu(b[0:lenF, :, :]).view([lenF, m[1], m[2]]) + 0.5  # minimum 0.5
    t = torch.arange(0.5, lenF * 1.0).view([lenF, 1, 1]).repeat([1, m[1], m[2]])
    t = t.to(aa.device)
    denom = (aa.lgamma().exp()) * (theta**aa)
    mid = t ** (aa - 1)
    right = torch.exp(-t / theta)
    w = 1 / denom * mid * right
    w = w / w.sum(0)  # scale to 1 for each UH

    return w

def change_param_range(param: torch.Tensor, bounds: list[float]) -> torch.Tensor:
    """Change the range of a parameter to the specified bounds.

    Parameters
    ----------
    param
        The parameter.
    bounds
        The parameter bounds.

    Returns
    -------
    torch.Tensor
        The parameter with the specified bounds.
    """
    return param * (bounds[1] - bounds[0]) + bounds[0]



def uh_conv(x, UH, viewmode=1):
    r"""Unit hydrograph convolution.

    UH is a vector indicating the unit hydrograph the convolved dimension will
    be the last dimension UH convolution is.
    Q(t)= integral(x(\tao)*UH(t-\tao))d\tao
    conv1d does
    integral(w(\tao)*x(t+\tao))d\tao
    hence we flip the UH
    https://programmer.group/pytorch-learning-conv1d-conv2d-and-conv3d.html
    view.

    x: [batch, var, time]
    UH:[batch, var, uhLen]
    batch needs to be accommodated by channels and we make use of groups
    https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
    https://pytorch.org/docs/stable/nn.functional.html
    """
    mm = x.shape
    nb = mm[0]
    m = UH.shape[-1]
    padd = m - 1
    if viewmode == 1:
        xx = x.view([1, nb, mm[-1]])
        w = UH.view([nb, 1, m])
        groups = nb

    y = F.conv1d(
        xx, torch.flip(w, [2]), groups=groups, padding=padd, stride=1, bias=None
    )
    if padd != 0:
        y = y[:, :, 0:-padd]
    return y.view(mm)


def source_flow_calculation(config, flow_out, c_NN, after_routing=True):
    """Source flow calculation."""
    varC_NN = config['var_c_nn']
    if 'DRAIN_SQKM' in varC_NN:
        area_name = 'DRAIN_SQKM'
    elif 'area_gages2' in varC_NN:
        area_name = 'area_gages2'
    else:
        print("area of basins are not available among attributes dataset")
    area = (
        c_NN[:, varC_NN.index(area_name)]
        .unsqueeze(0)
        .unsqueeze(-1)
        .repeat(flow_out['flow_sim'].shape[0], 1, 1)
    )
    # flow calculation. converting mm/day to m3/sec
    if after_routing:
        srflow = (
            (1000 / 86400) * area * (flow_out['srflow']).repeat(1, 1, config['nmul'])
        )  # Q_t - gw - ss
        ssflow = (
            (1000 / 86400) * area * (flow_out['ssflow']).repeat(1, 1, config['nmul'])
        )  # ras
        gwflow = (
            (1000 / 86400) * area * (flow_out['gwflow']).repeat(1, 1, config['nmul'])
        )
    else:
        srflow = (
            (1000 / 86400)
            * area
            * (flow_out['srflow_no_rout']).repeat(1, 1, config['nmul'])
        )  # Q_t - gw - ss
        ssflow = (
            (1000 / 86400)
            * area
            * (flow_out['ssflow_no_rout']).repeat(1, 1, config['nmul'])
        )  # ras
        gwflow = (
            (1000 / 86400)
            * area
            * (flow_out['gwflow_no_rout']).repeat(1, 1, config['nmul'])
        )
    # srflow = torch.clamp(srflow, min=0.0)  # to remove the small negative values
    # ssflow = torch.clamp(ssflow, min=0.0)
    # gwflow = torch.clamp(gwflow, min=0.0)
    return srflow, ssflow, gwflow


class Hbv_2(torch.nn.Module):
    """HBV 2.0 ~.

    Multi-component, multiscale, differentiable PyTorch HBV model with rainfall
    runoff simulation on unit basins.

    Authors
    -------
    -   Yalan Song, Leo Lonzarich
    -   (Original NumPy HBV ver.) Beck et al., 2020 (http://www.gloh2o.org/hbv/).
    -   (HBV-light Version 2) Seibert, 2005
        (https://www.geo.uzh.ch/dam/jcr:c8afa73c-ac90-478e-a8c7-929eed7b1b62/HBV_manual_2005.pdf).

    Publication
    -----------
    -   Yalan Song, Tadd Bindas, Chaopeng Shen, et al. "High-resolution
        national-scale water modeling is enhanced by multiscale differentiable
        physics-informed machine learning." Water Resources Research (2025).
        https://doi.org/10.1029/2024WR038928.

    Parameters
    ----------
    config
        Configuration dictionary.
    device
        Device to run the model on.
    """

    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.name = 'HBV 2.0'
        self.config = config
        self.initialize = False
        self.warm_up = 0
        self.pred_cutoff = 0
        self.warm_up_states = True
        self.dynamic_params = []
        self.dy_drop = 0.0
        self.variables = ['prcp', 'tmean', 'pet']
        self.routing = True
        self.comprout = False
        self.nearzero = 1e-5
        self.nmul = 1
        self.cache_states = False
        self.device = device

        self.states, self._states_cache = None, None

        self.state_names = [
            'SNOWPACK',  # Snowpack storage
            'MELTWATER',  # Meltwater storage
            'SM',  # Soil moisture storage
            'SUZ',  # Upper groundwater storage
            'SLZ',  # Lower groundwater storage
        ]
        self.flux_names = [
            'streamflow',  # Routed Streamflow
            'srflow',  # Routed surface runoff
            'ssflow',  # Routed subsurface flow
            'gwflow',  # Routed groundwater flow
            'AET_hydro',  # Actual ET
            'PET_hydro',  # Potential ET
            'SWE',  # Snow water equivalent
            'streamflow_no_rout',  # Streamflow
            'srflow_no_rout',  # Surface runoff
            'ssflow_no_rout',  # Subsurface flow
            'gwflow_no_rout',  # Groundwater flow
            'recharge',  # Recharge
            'excs',  # Excess stored water
            'evapfactor',  # Evaporation factor
            'tosoil',  # Infiltration
            'percolation',  # Percolation
            'capillary',  # Capillary rise
            'BFI',  # Baseflow index
        ]

        self.parameter_bounds = {
            'parBETA': [1.0, 6.0],
            'parFC': [50, 1000],
            'parK0': [0.05, 0.9],
            'parK1': [0.01, 0.5],
            'parK2': [0.001, 0.2],
            'parLP': [0.2, 1],
            'parPERC': [0, 10],
            'parUZL': [0, 100],
            'parTT': [-2.5, 2.5],
            'parCFMAX': [0.5, 10],
            'parCFR': [0, 0.1],
            'parCWH': [0, 0.2],
            'parBETAET': [0.3, 5],
            'parC': [0, 1],
            'parRT': [0, 20],
            'parAC': [0, 2500],
        }
        self.routing_parameter_bounds = {
            'route_a': [0, 2.9],
            'route_b': [0, 6.5],
        }

        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if config is not None:
            # Overwrite defaults with config values.
            self.warm_up = config.get('warm_up', self.warm_up)
            self.warm_up_states = config.get('warm_up_states', self.warm_up_states)
            self.dy_drop = config.get('dy_drop', self.dy_drop)
            self.dynamic_params = config['dynamic_params'].get(
                self.__class__.__name__, self.dynamic_params
            )
            self.variables = config.get('variables', self.variables)
            self.routing = config.get('routing', self.routing)
            self.comprout = config.get('comprout', self.comprout)
            self.nearzero = config.get('nearzero', self.nearzero)
            self.nmul = config.get('nmul', self.nmul)
            self.cache_states = config.get('cache_states', False)
        self._set_parameters()

    def _init_states(self, ngrid: int) -> tuple[torch.Tensor]:
        """Initialize model states to zero."""

        def make_state():
            return torch.full(
                (ngrid, self.nmul), 0.001, dtype=torch.float32, device=self.device
            )

        return tuple(make_state() for _ in range(len(self.state_names)))

    def get_states(self) -> Optional[tuple[torch.Tensor, ...]]:
        """Return internal model states.

        Returns
        -------
        tuple[torch.Tensor, ...]
            A tuple containing the states (SNOWPACK, MELTWATER, SM, SUZ, SLZ).
        """
        return self._states_cache

    def load_states(
        self,
        states: tuple[torch.Tensor, ...],
    ) -> None:
        """Load internal model states and set to model device and type.

        Parameters
        ----------
        states
            A tuple containing the states (SNOWPACK, MELTWATER, SM, SUZ, SLZ).
        """
        for state in states:
            if not isinstance(state, torch.Tensor):
                raise ValueError("Each element in `states` must be a tensor.")
        nstates = len(self.state_names)
        if not (isinstance(states, tuple) and len(states) == nstates):
            raise ValueError(f"`states` must be a tuple of {nstates} tensors.")

        self.states = tuple(
            s.detach().to(self.device, dtype=torch.float32) for s in states
        )

    def _set_parameters(self) -> None:
        """Get physical parameters."""
        self.phy_param_names = self.parameter_bounds.keys()
        if self.routing:
            self.routing_param_names = self.routing_parameter_bounds.keys()
        else:
            self.routing_param_names = []

        self.learnable_param_count1 = len(self.dynamic_params) * self.nmul
        self.learnable_param_count2 = (
            len(self.phy_param_names) - len(self.dynamic_params)
        ) * self.nmul + len(self.routing_param_names)
        self.learnable_param_count = (
            self.learnable_param_count1 + self.learnable_param_count2
        )

    def _unpack_parameters(
        self,
        parameters: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract physical model and routing parameters from NN output.

        Parameters
        ----------
        parameters
            Unprocessed, learned parameters from a neural network.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple of physical and routing parameters.
        """
        phy_param_count = len(self.parameter_bounds)
        dy_param_count = len(self.dynamic_params)
        dif_count = phy_param_count - dy_param_count

        # Physical dynamic parameters
        phy_dy_params = parameters[0].view(
            parameters[0].shape[0],
            parameters[0].shape[1],
            dy_param_count,
            self.nmul,
        )

        # Physical static parameters
        phy_static_params = parameters[1][:, : dif_count * self.nmul].view(
            parameters[1].shape[0],
            dif_count,
            self.nmul,
        )

        # Routing parameters
        routing_params = None
        if self.routing:
            routing_params = parameters[1][:, dif_count * self.nmul :]

        return (phy_dy_params, phy_static_params, routing_params)

    def _descale_phy_dy_parameters(
        self,
        phy_dy_params: torch.Tensor,
        dy_list: list,
    ) -> dict[str, torch.Tensor]:
        """Descale physical parameters.

        Parameters
        ----------
        phy_params
            Normalized physical parameters.
        dy_list
            List of dynamic parameters.

        Returns
        -------
        dict
            Dictionary of descaled physical parameters.
        """
        nsteps = phy_dy_params.shape[0]
        ngrid = phy_dy_params.shape[1]

        # TODO: Fix; if dynamic parameters are not entered in config as they are
        # in HBV params list, then descaling misamtch will occur.
        param_dict = {}
        pmat = torch.ones([1, ngrid, 1]) * self.dy_drop
        for i, name in enumerate(dy_list):
            staPar = phy_dy_params[-1, :, i, :].unsqueeze(0).repeat([nsteps, 1, 1])

            dynPar = phy_dy_params[:, :, i, :]
            drmask = torch.bernoulli(pmat).detach_().to(self.device)

            comPar = dynPar * (1 - drmask) + staPar * drmask
            param_dict[name] = change_param_range(
                param=comPar,
                bounds=self.parameter_bounds[name],
            )
        return param_dict

    def _descale_phy_stat_parameters(
        self,
        phy_stat_params: torch.Tensor,
        stat_list: list,
    ) -> torch.Tensor:
        """Descale routing parameters.

        Parameters
        ----------
        routing_params
            Normalized routing parameters.

        Returns
        -------
        dict
            Dictionary of descaled routing parameters.
        """
        parameter_dict = {}
        for i, name in enumerate(stat_list):
            param = phy_stat_params[:, i, :]

            parameter_dict[name] = change_param_range(
                param=param,
                bounds=self.parameter_bounds[name],
            )
        return parameter_dict

    def _descale_route_parameters(
        self,
        routing_params: torch.Tensor,
    ) -> torch.Tensor:
        """Descale routing parameters.

        Parameters
        ----------
        routing_params
            Normalized routing parameters.

        Returns
        -------
        dict
            Dictionary of descaled routing parameters.
        """
        parameter_dict = {}
        for i, name in enumerate(self.routing_parameter_bounds.keys()):
            param = routing_params[:, i]

            parameter_dict[name] = change_param_range(
                param=param,
                bounds=self.routing_parameter_bounds[name],
            )
        return parameter_dict

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        parameters: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], tuple]:
        """Forward pass.

        Parameters
        ----------
        x_dict
            Dictionary of input forcing data.
        parameters
            Unprocessed, learned parameters from a neural network.

        Returns
        -------
        tuple[dict, tuple]
            Tuple or dictionary of model outputs.
        """
        # Unpack input data.
        x = x_dict['x_phy']
        Ac = x_dict['ac_all'].unsqueeze(-1).repeat(1, self.nmul)
        Elevation = x_dict['elev_all'].unsqueeze(-1).repeat(1, self.nmul)
        self.muwts = x_dict.get('muwts', None)
        ngrid = x.shape[1]

        # Unpack parameters.
        phy_dy_params, phy_static_params, routing_params = self._unpack_parameters(
            parameters
        )

        if self.routing:
            self.routing_param_dict = self._descale_route_parameters(routing_params)
        phy_dy_params_dict = self._descale_phy_dy_parameters(
            phy_dy_params,
            dy_list=self.dynamic_params,
        )
        phy_static_params_dict = self._descale_phy_stat_parameters(
            phy_static_params,
            stat_list=[
                param
                for param in self.phy_param_names
                if param not in self.dynamic_params
            ],
        )

        if (not self.states) or (not self.cache_states):
            current_states = self._init_states(ngrid)
        else:
            current_states = self.states

        fluxes, states = self._PBM(
            x,
            Ac,
            Elevation,
            current_states,
            phy_dy_params_dict,
            phy_static_params_dict,
        )

        # State caching
        self._state_cache = [s.detach() for s in states]

        if self.cache_states:
            self.states = self._state_cache

        return fluxes

    def _PBM(
        self,
        forcing: torch.Tensor,
        Ac: torch.Tensor,
        Elevation: torch.Tensor,
        states: tuple,
        phy_dy_params_dict: dict,
        phy_static_params_dict: dict,
    ) -> Union[tuple, dict[str, torch.Tensor]]:
        """Run through process-based model (PBM).

        Parameters
        ----------
        forcing
            Input forcing data.
        states
            Initial model states.
        full_param_dict
            Dictionary of model parameters.

        Returns
        -------
        Union[tuple, dict]
            Tuple or dictionary of model outputs.
        """
        SNOWPACK, MELTWATER, SM, SUZ, SLZ = states

        # Forcings
        P = forcing[:, :, self.variables.index('prcp')]  # Precipitation
        T = forcing[:, :, self.variables.index('tmean')]  # Mean air temp
        PET = forcing[:, :, self.variables.index('pet')]  # Potential ET
        nsteps, ngrid = P.shape

        # Expand dims to accomodate for nmul models.
        Pm = P.unsqueeze(2).repeat(1, 1, self.nmul)
        Tm = T.unsqueeze(2).repeat(1, 1, self.nmul)
        PETm = PET.unsqueeze(-1).repeat(1, 1, self.nmul)

        # Apply correction factor to precipitation
        # P = parPCORR.repeat(nsteps, 1) * P

        # Initialize time series of model variables in shape [time, basins, nmul].
        Qsimmu = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device) + 0.001
        Q0_sim = (
            torch.zeros(Pm.size(), dtype=torch.float32, device=self.device) + 0.0001
        )
        Q1_sim = (
            torch.zeros(Pm.size(), dtype=torch.float32, device=self.device) + 0.0001
        )
        Q2_sim = (
            torch.zeros(Pm.size(), dtype=torch.float32, device=self.device) + 0.0001
        )

        AET = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        recharge_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        excs_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        evapfactor_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        tosoil_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        PERC_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        SWE_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        capillary_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)

        param_dict = {}
        for t in range(nsteps):
            # Get dynamic parameter values per timestep.
            for key in phy_dy_params_dict.keys():
                param_dict[key] = phy_dy_params_dict[key][t, :, :]
            for key in phy_static_params_dict.keys():
                param_dict[key] = phy_static_params_dict[key][:, :]

            # Separate precipitation into liquid and solid components.
            PRECIP = Pm[t, :, :]
            parTT_new = (Elevation >= 2000).type(torch.float32) * 4.0 + (
                Elevation < 2000
            ).type(torch.float32) * param_dict['parTT']
            RAIN = torch.mul(PRECIP, (Tm[t, :, :] >= parTT_new).type(torch.float32))
            SNOW = torch.mul(PRECIP, (Tm[t, :, :] < parTT_new).type(torch.float32))

            # Snow -------------------------------
            SNOWPACK = SNOWPACK + SNOW
            melt = param_dict['parCFMAX'] * (Tm[t, :, :] - parTT_new)
            # melt[melt < 0.0] = 0.0
            melt = torch.clamp(melt, min=0.0)
            # melt[melt > SNOWPACK] = SNOWPACK[melt > SNOWPACK]
            melt = torch.min(melt, SNOWPACK)
            MELTWATER = MELTWATER + melt
            SNOWPACK = SNOWPACK - melt
            refreezing = (
                param_dict['parCFR']
                * param_dict['parCFMAX']
                * (parTT_new - Tm[t, :, :])
            )
            # refreezing[refreezing < 0.0] = 0.0
            # refreezing[refreezing > MELTWATER] = MELTWATER[refreezing > MELTWATER]
            refreezing = torch.clamp(refreezing, min=0.0)
            refreezing = torch.min(refreezing, MELTWATER)
            SNOWPACK = SNOWPACK + refreezing
            MELTWATER = MELTWATER - refreezing
            tosoil = MELTWATER - (param_dict['parCWH'] * SNOWPACK)
            tosoil = torch.clamp(tosoil, min=0.0)
            MELTWATER = MELTWATER - tosoil

            # Soil and evaporation -------------------------------
            soil_wetness = (SM / param_dict['parFC']) ** param_dict['parBETA']
            # soil_wetness[soil_wetness < 0.0] = 0.0
            # soil_wetness[soil_wetness > 1.0] = 1.0
            soil_wetness = torch.clamp(soil_wetness, min=0.0, max=1.0)
            recharge = (RAIN + tosoil) * soil_wetness

            SM = SM + RAIN + tosoil - recharge

            excess = SM - param_dict['parFC']
            excess = torch.clamp(excess, min=0.0)
            SM = SM - excess
            # NOTE: Different from HBV 1.0. Add static/dynamicET shape parameter parBETAET.
            evapfactor = (
                SM / (param_dict['parLP'] * param_dict['parFC'])
            ) ** param_dict['parBETAET']
            evapfactor = torch.clamp(evapfactor, min=0.0, max=1.0)
            ETact = PETm[t, :, :] * evapfactor
            ETact = torch.min(SM, ETact)
            SM = torch.clamp(SM - ETact, min=self.nearzero)

            # Capillary rise (HBV 1.1p mod) -------------------------------
            capillary = torch.min(
                SLZ,
                param_dict['parC']
                * SLZ
                * (1.0 - torch.clamp(SM / param_dict['parFC'], max=1.0)),
            )

            SM = torch.clamp(SM + capillary, min=self.nearzero)
            SLZ = torch.clamp(SLZ - capillary, min=self.nearzero)

            # Groundwater boxes -------------------------------
            SUZ = SUZ + recharge + excess
            PERC = torch.min(SUZ, param_dict['parPERC'])
            SUZ = SUZ - PERC
            Q0 = param_dict['parK0'] * torch.clamp(SUZ - param_dict['parUZL'], min=0.0)
            SUZ = SUZ - Q0
            Q1 = param_dict['parK1'] * SUZ
            SUZ = SUZ - Q1
            SLZ = SLZ + PERC

            LF = torch.clamp(
                (Ac - param_dict['parAC']) / 1000, min=-1, max=1
            ) * param_dict['parRT'] * (Ac < 2500) + torch.exp(
                torch.clamp(-(Ac - 2500) / 50, min=-10.0, max=0.0)
            ) * param_dict['parRT'] * (Ac >= 2500)
            SLZ = torch.clamp(SLZ + LF, min=0.0)

            Q2 = param_dict['parK2'] * SLZ
            SLZ = SLZ - Q2

            Qsimmu[t, :, :] = Q0 + Q1 + Q2
            Q0_sim[t, :, :] = Q0
            Q1_sim[t, :, :] = Q1
            Q2_sim[t, :, :] = Q2
            AET[t, :, :] = ETact
            SWE_sim[t, :, :] = SNOWPACK
            capillary_sim[t, :, :] = capillary

            recharge_sim[t, :, :] = recharge
            excs_sim[t, :, :] = excess
            evapfactor_sim[t, :, :] = evapfactor
            tosoil_sim[t, :, :] = tosoil
            PERC_sim[t, :, :] = PERC

        # Get the average or weighted average using learned weights.
        if self.muwts is None:
            Qsimavg = Qsimmu.mean(-1)
        else:
            Qsimavg = (Qsimmu * self.muwts).sum(-1)

        # Run routing
        if self.routing:
            # Routing for all components or just the average.
            if self.comprout:
                # All components; reshape to [time, gages * num models]
                Qsim = Qsimmu.view(nsteps, ngrid * self.nmul)
            else:
                # Average, then do routing.
                Qsim = Qsimavg

            UH = uh_gamma(
                self.routing_param_dict['route_a'].repeat(nsteps, 1).unsqueeze(-1),
                self.routing_param_dict['route_b'].repeat(nsteps, 1).unsqueeze(-1),
                lenF=15,
            )
            rf = torch.unsqueeze(Qsim, -1).permute([1, 2, 0])  # [gages,vars,time]
            UH = UH.permute([1, 2, 0])  # [gages,vars,time]
            Qsrout = uh_conv(rf, UH).permute([2, 0, 1])

            # Routing individually for Q0, Q1, and Q2, all w/ dims [gages,vars,time].
            rf_Q0 = Q0_sim.mean(-1, keepdim=True).permute([1, 2, 0])
            Q0_rout = uh_conv(rf_Q0, UH).permute([2, 0, 1])
            rf_Q1 = Q1_sim.mean(-1, keepdim=True).permute([1, 2, 0])
            Q1_rout = uh_conv(rf_Q1, UH).permute([2, 0, 1])
            rf_Q2 = Q2_sim.mean(-1, keepdim=True).permute([1, 2, 0])
            Q2_rout = uh_conv(rf_Q2, UH).permute([2, 0, 1])

            if self.comprout:
                # Qs is now shape [time, [gages*num models], vars]
                Qstemp = Qsrout.view(nsteps, ngrid, self.nmul)
                if self.muwts is None:
                    Qs = Qstemp.mean(-1, keepdim=True)
                else:
                    Qs = (Qstemp * self.muwts).sum(-1, keepdim=True)
            else:
                Qs = Qsrout

        else:
            # No routing, only output the average of all model sims.
            Qs = torch.unsqueeze(Qsimavg, -1)
            Q0_rout = Q1_rout = Q2_rout = None

        states = (SNOWPACK, MELTWATER, SM, SUZ, SLZ)

        if self.initialize:
            # If initialize is True, only return warmed-up storages.
            return states
        else:
            # Baseflow index (BFI) calculation
            BFI_sim = (
                100
                * (torch.sum(Q2_rout, dim=0) / (torch.sum(Qs, dim=0) + self.nearzero))[
                    :, 0
                ]
            )

            # Return all sim results.
            flux_dict = {
                'streamflow': Qs,  # Routed Streamflow
                'srflow': Q0_rout,  # Routed surface runoff
                'ssflow': Q1_rout,  # Routed subsurface flow
                'gwflow': Q2_rout,  # Routed groundwater flow
                'AET_hydro': AET.mean(-1, keepdim=True),  # Actual ET
                'PET_hydro': PETm.mean(-1, keepdim=True),  # Potential ET
                'SWE': SWE_sim.mean(-1, keepdim=True),  # Snow water equivalent
                'streamflow_no_rout': Qsim.unsqueeze(dim=2),  # Streamflow
                'srflow_no_rout': Q0_sim.mean(-1, keepdim=True),  # Surface runoff
                'ssflow_no_rout': Q1_sim.mean(-1, keepdim=True),  # Subsurface flow
                'gwflow_no_rout': Q2_sim.mean(-1, keepdim=True),  # Groundwater flow
                'recharge': recharge_sim.mean(-1, keepdim=True),  # Recharge
                'excs': excs_sim.mean(-1, keepdim=True),  # Excess stored water
                'evapfactor': evapfactor_sim.mean(
                    -1, keepdim=True
                ),  # Evaporation factor
                'tosoil': tosoil_sim.mean(-1, keepdim=True),  # Infiltration
                'percolation': PERC_sim.mean(-1, keepdim=True),  # Percolation
                'capillary': capillary_sim.mean(-1, keepdim=True),  # Capillary rise
                'BFI': BFI_sim,  # Baseflow index
            }

            if not self.warm_up_states:
                for key in flux_dict.keys():
                    if key != 'BFI':
                        flux_dict[key] = flux_dict[key][self.pred_cutoff :, :, :]
            return flux_dict, states

def HBV(meteo_data, parameters, years_init):
    
    '''    
    HBV(meteo_data, parameters, years_init)
    
    Runs the HBV hydrological model in a spatially-distributed fashion. No NaNs
    allowed in the input.
    
    Inputs:
        meteo_data = Dict with fields P, Temp, and ETpot. Each field is a two-
            dimensional array with daily values in mm/d. The rows represent the
            time steps and the columns represent the grid-cells.
        parameters = Dict with one-dimensional model parameter arrays. Each 
            value in the array represents a different grid-cell. 
        years_init = Number of years to run the model to initialize the stores
            before starting the real run.
            
    Output:
        Qsim = Daily values of simulated streamflow in mm/d.
    '''
    
    # Initialize time series of model variables
    SNOWPACK = np.zeros(parameters['BETA'].shape,dtype=np.float32)+0.001
    MELTWATER = np.zeros(parameters['BETA'].shape,dtype=np.float32)+0.001
    SM = np.zeros(parameters['BETA'].shape,dtype=np.float32)+0.001
    SUZ = np.zeros(parameters['BETA'].shape,dtype=np.float32)+0.001
    SLZ = np.zeros(parameters['BETA'].shape,dtype=np.float32)+0.001
    ETact = np.zeros(parameters['BETA'].shape,dtype=np.float32)+0.001
    Qsim = np.zeros(meteo_data['P'].shape,dtype=np.float32)*np.NaN
    Qsim[0,:] = 0.001
    
    # Start loop
    init_days = np.min((int(years_init*365),meteo_data['P'].shape[0]-1))
    time_step = 0
    init_day_counter = 0
    init_done = False
    while time_step<meteo_data['P'].shape[0]:
        
        # Separate precipitation into liquid and solid components
        PRECIP = meteo_data['P'][time_step,:]*parameters['PCORR']
        RAIN = np.multiply(PRECIP,meteo_data['Temp'][time_step,:]>=parameters['TT'])
        SNOW = np.multiply(PRECIP,meteo_data['Temp'][time_step,:]<parameters['TT'])
        SNOW = SNOW*parameters['SFCF']
        
        # Snow
        SNOWPACK = SNOWPACK+SNOW
        melt = parameters['CFMAX']*(meteo_data['Temp'][time_step,:]-parameters['TT'])
        melt = melt.clip(0,SNOWPACK)
        MELTWATER = MELTWATER+melt
        SNOWPACK = SNOWPACK-melt
        refreezing = parameters['CFR']*parameters['CFMAX'] * (parameters['TT']-meteo_data['Temp'][time_step,:])
        refreezing = refreezing.clip(0,MELTWATER)
        SNOWPACK = SNOWPACK+refreezing
        MELTWATER = MELTWATER-refreezing
        tosoil = MELTWATER-(parameters['CWH']*SNOWPACK)
        tosoil = tosoil.clip(0,None)
        MELTWATER = MELTWATER-tosoil

        # Soil and evaporation
        soil_wetness = (SM/parameters['FC']) ** parameters['BETA']
        soil_wetness = soil_wetness.clip(0,1.0)
        recharge = (RAIN+tosoil) * soil_wetness
        SM = SM+RAIN+tosoil-recharge
        excess = SM-parameters['FC']
        excess = excess.clip(0,None)
        SM = SM-excess
        evapfactor = SM/(parameters['LP']*parameters['FC'])
        evapfactor = evapfactor.clip(0,1.0)
        ETact = meteo_data['ETpot'][time_step,:]*evapfactor
        ETact = np.minimum(SM, ETact)
        SM = SM-ETact

        # Groundwater boxes
        SUZ = SUZ+recharge+excess
        PERC = np.minimum(SUZ, parameters['PERC'])
        SUZ = SUZ-PERC
        Q0 = parameters['K0']*np.maximum(SUZ-parameters['UZL'], 0.0)
        SUZ = SUZ-Q0
        Q1 = parameters['K1']*SUZ
        SUZ = SUZ-Q1
        SLZ = SLZ+PERC
        Q2 = parameters['K2']*SLZ
        SLZ = SLZ-Q2
        Qsim[time_step,:] = Q0+Q1+Q2
        
        time_step = time_step+1
        init_day_counter = init_day_counter+1
                
        # Go back to date_start once we've reached the initialization period
        if (init_done==False) & (init_day_counter==init_days): 
            time_step = 0
            init_done = True

    return Qsim






from typing import Dict, List, Optional, Tuple, Union

import torch

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


class BaseConceptualModel(nn.Module):
    """Abstract base model class

    Don't use this class for model training!The purpose is to have some common operations that all conceptual models
    will need.

    """

    def __init__(
        self,
    ):
        super(BaseConceptualModel, self).__init__()

    def forward(
        self,
        x_conceptual: torch.Tensor,
        parameters: Dict[str, torch.Tensor],
        initial_states: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        raise NotImplementedError

    def map_parameters(
        self, lstm_out: torch.Tensor, warmup_period: int
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Map output of data-driven part to predefined ranges of the conceptual model parameters.

        The result are two dictionaries, one contains the parameters for the warmup period of the conceptual model and
        the other contains the parameters for the simulation period. Moreover, the parameterization can be static or
        dynamic. In the static parameterization the last value is repeated over the whole timeseries, while in the
        dynamic parameterization we have one parameter set for each time step.

        Note:
            The dynamic parameterization only occurs in the simulation phase, not the warmup! The warmup always uses
            static parameterization. Therefore, in case we specified dynamic parameterization, for the warmup period,
            we take the last value of this period and repeat it throughout the warmup phase.

        Parameters
        ----------
        lstm_out : torch.Tensor
            Tensor of size [batch_size, time_steps, n_param] that will be mapped to the predefined ranges of the
            conceptual model parameters to act as the dynamic parameterization.
        warmup_period : int
            Number of timesteps (e.g. days) to warmup the internal states of the conceptual model

        Returns
        -------
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
            - parameters_warmup : Dict[str, torch.Tensor]
                Parameters for the warmup period (always static!)
            - parameters_simulation : Dict[str, torch.Tensor]
                Parameterization of the conceptual model in the training/testing period. Can be static or dynamic

        """
        # Reshape tensor to consider multiple conceptual models running in parallel.
        lstm_out = lstm_out.view(lstm_out.shape[0], lstm_out.shape[1], -1, self.n_conceptual_models)

        parameters_warmup = {}
        parameters_simulation = {}
        for index, (parameter_name, parameter_range) in enumerate(self.parameter_ranges.items()):
            range_t = torch.tensor(parameter_range, dtype=torch.float32, device=lstm_out.device)
            range_t = range_t.unsqueeze(dim=1).unsqueeze(dim=2)

            if self.parameter_type[parameter_name] == "static":
                # If parameter is static, take the last value predicted by the lstm and copy it for all the timesteps.
                warmup_lstm_out = lstm_out[:, -1:, index, :].expand(-1, warmup_period, -1)
                simulation_lstm_out = lstm_out[:, -1:, index, :].expand(-1, lstm_out.shape[1] - warmup_period, -1)
            elif self.parameter_type[parameter_name] == "dynamic":
                warmup_lstm_out = lstm_out[:, warmup_period - 1 : warmup_period, index, :].expand(-1, warmup_period, -1)
                simulation_lstm_out = lstm_out[:, warmup_period:, index, :]
            else:
                raise ValueError(f"Unsupported parameter type {self.parameter_type[parameter_name]}")

            parameters_warmup[parameter_name] = range_t[:1, :, :] + torch.sigmoid(warmup_lstm_out) * (
                range_t[1:, :, :] - range_t[:1, :, :]
            )

            parameters_simulation[parameter_name] = range_t[:1, :, :] + torch.sigmoid(simulation_lstm_out) * (
                range_t[1:, :, :] - range_t[:1, :, :]
            )

        return parameters_warmup, parameters_simulation

    def _map_parameter_type(self, parameter_type: List[str] = None):
        """Define parameter type, static or dynamic.

        The model parameters can be static or dynamic. This function creates a dictionary that associate the parameter
        name with a type specified by the user. In case the user did not specify a type, the parameter_type is
        automatically specified as static.

        Parameters
        ----------
        parameter_type : List[str]
            List to specify which parameters of the conceptual model will be dynamic.

        Returns
        -------
        map_parameter_type: Dict[str, str]
            Dictionary

        """
        map_parameter_type = {}
        for key, _ in self.parameter_ranges.items():
            if parameter_type is not None and key in parameter_type:  # if user specified the type
                map_parameter_type[key] = "dynamic"
            else:  # default initialization
                map_parameter_type[key] = "static"

        return map_parameter_type

    def _initialize_information(self, conceptual_inputs: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Initialize structures to store the time evolution of the internal states and the outflow

        Parameters
        ----------
        conceptual_inputs: torch.Tensor
            Inputs of the conceptual model

        Returns
        -------
        Tuple[Dict[str, torch.Tensor], torch.Tensor]
            - states: Dict[str, torch.Tensor]
                Dictionary to store the time evolution of the internal states (buckets) of the conceptual model
            - q_out: torch.Tensor
                Tensor to store the outputs of the conceptual model

        """
        states = {}
        # initialize dictionary to store the evolution of the states
        for name, _ in self._initial_states.items():
            states[name] = torch.zeros(
                (conceptual_inputs.shape[0], conceptual_inputs.shape[1], self.n_conceptual_models),
                dtype=torch.float32,
                device=conceptual_inputs.device,
            )

        # initialize vectors to store the evolution of the outputs
        out = torch.zeros(
            (conceptual_inputs.shape[0], conceptual_inputs.shape[1], self.output_size),
            dtype=torch.float32,
            device=conceptual_inputs.device,
        )

        return states, out

    def _get_final_states(self, states: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Recovers final states of the conceptual model.

        Parameters
        ----------
        states : Dict[str, torch.Tensor]
            Dictionary with the time evolution of the internal states (buckets) of the conceptual model

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary with the internal states (buckets) of the conceptual model, on the last timestep

        """
        return {name: state[:, -1, :] for name, state in states.items()}

    @property
    def _initial_states(self) -> Dict[str, float]:
        raise NotImplementedError

    @property
    def parameter_ranges(self) -> Dict[str, List[float]]:
        raise NotImplementedError


class HBV(BaseConceptualModel):
    """HBV model.

    Implementation based on Feng et al. [1]_ and Seibert [2]_. The code creates a modified version of the HBV model
    that can be used as a differentiable entity to create hybrid models. One can run multiple entities of the model at
    the same time.

    Parameters
    ----------
    n_models : int
        Number of model entities that will be run at the same time
    parameter_type : List[str]
        List to specify which parameters of the conceptual model will be dynamic.

    References
    ----------
    .. [1] Feng, D., Liu, J., Lawson, K., & Shen, C. (2022). Differentiable, learnable, regionalized process-based
        models with multiphysical outputs can approach state-of-the-art hydrologic prediction accuracy. Water Resources
        Research, 58, e2022WR032404. https://doi.org/10.1029/2022WR032404
    .. [2] Seibert, J. (2005) HBV Light Version 2. User’s Manual. Department of Physical Geography and Quaternary
        Geology, Stockholm University, Stockholm
    
    """

    def __init__(self, n_models: int = 1, parameter_type: List[str] = None):
        super(HBV, self).__init__()
        self.n_conceptual_models = n_models
        self.parameter_type = self._map_parameter_type(parameter_type=parameter_type)
        self.output_size = 1

    def forward(
        self,
        x_conceptual: torch.Tensor,
        parameters: Dict[str, torch.Tensor],
        initial_states: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass on the HBV model.

        Parameters
        ----------
        x_conceptual: torch.Tensor
            Tensor of size [batch_size, time_steps, n_inputs]. The batch_size is associated with a certain basin and a
            certain prediction period. The time_steps refer to the number of time steps (e.g. days) that our conceptual
            model is going to be run for. The n_inputs refer to the dynamic forcings used to run the conceptual model
            (e.g. Precipitation, Temperature...)
        parameter_type : List[str]
            List to specify which parameters of the conceptual model will be dynamic.
        initial_states: Optional[Dict[str, torch.Tensor]]
            Optional parameter! In case one wants to specify the initial state of the internal states of the conceptual
            model.

        Returns
        -------
        Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]
            y_hat: torch.Tensor
                Simulated outflow
            parameters: Dict[str, torch.Tensor]
                Dynamic parameterization of the conceptual model
            internal_states: Dict[str, torch.Tensor]
                Time-evolution of the internal states of the conceptual model
            last_states: Dict[str, torch.Tensor]
                Internal states of the conceptual model in the last timestep

        """
        # initialize structures to store the information
        states, out = self._initialize_information(conceptual_inputs=x_conceptual)

        # initialize constants
        zero = torch.tensor(0.0, dtype=torch.float32, device=x_conceptual.device)

        # Broadcast tensor to consider multiple conceptual models running in parallel
        precipitation = torch.tile(x_conceptual[:, :, 0].unsqueeze(2), (1, 1, self.n_conceptual_models))
        et = torch.tile(x_conceptual[:, :, 1].unsqueeze(2), (1, 1, self.n_conceptual_models))
        if x_conceptual.shape[2] == 4:  # the user specified tmax and tmin
            temperature = (x_conceptual[:, :, 2] + x_conceptual[:, :, 3]) / 2
        else:
            temperature = x_conceptual[:, :, 2]
        temperature = torch.tile(temperature.unsqueeze(2), (1, 1, self.n_conceptual_models))

        # Division between solid and liquid precipitation can be done outside of the loop
        temp_mask = temperature < parameters["TT"]
        liquid_p = precipitation.clone()
        liquid_p[temp_mask] = zero
        snow = precipitation.clone()
        snow[~temp_mask] = zero

        if initial_states is None:  # if we did not specify initial states it takes the default values
            SNOWPACK = torch.full(
                (x_conceptual.shape[0], self.n_conceptual_models),
                self._initial_states["SNOWPACK"],
                dtype=torch.float32,
                device=x_conceptual.device,
            )
            MELTWATER = torch.full(
                (x_conceptual.shape[0], self.n_conceptual_models),
                self._initial_states["MELTWATER"],
                dtype=torch.float32,
                device=x_conceptual.device,
            )
            SM = torch.full(
                (x_conceptual.shape[0], self.n_conceptual_models),
                self._initial_states["SM"],
                dtype=torch.float32,
                device=x_conceptual.device,
            )
            SUZ = torch.full(
                (x_conceptual.shape[0], self.n_conceptual_models),
                self._initial_states["SUZ"],
                dtype=torch.float32,
                device=x_conceptual.device,
            )
            SLZ = torch.full(
                (x_conceptual.shape[0], self.n_conceptual_models),
                self._initial_states["SLZ"],
                dtype=torch.float32,
                device=x_conceptual.device,
            )
        else:  # we specify the initial states
            SNOWPACK = initial_states["SNOWPACK"]
            MELTWATER = initial_states["MELTWATER"]
            SM = initial_states["SM"]
            SUZ = initial_states["SUZ"]
            SLZ = initial_states["SLZ"]

        # run hydrological model for each time step
        for j in range(x_conceptual.shape[1]):
            # Snow module -----------------------------------------------------------------------------------------
            SNOWPACK = SNOWPACK + snow[:, j, :]
            melt = parameters["CFMAX"][:, j, :] * (temperature[:, j, :] - parameters["TT"][:, j, :])
            melt = torch.clamp(melt, min=0.0)
            melt = torch.min(melt, SNOWPACK)
            MELTWATER = MELTWATER + melt
            SNOWPACK = SNOWPACK - melt
            refreezing = (
                parameters["CFR"][:, j, :]
                * parameters["CFMAX"][:, j, :]
                * (parameters["TT"][:, j, :] - temperature[:, j, :])
            )
            refreezing = torch.clamp(refreezing, min=0.0)
            refreezing = torch.min(refreezing, MELTWATER)
            SNOWPACK = SNOWPACK + refreezing
            MELTWATER = MELTWATER - refreezing
            tosoil = MELTWATER - (parameters["CWH"][:, j, :] * SNOWPACK)
            tosoil = torch.clamp(tosoil, min=0.0)
            MELTWATER = MELTWATER - tosoil

            # Soil and evaporation ---------------------------------------------------------------------------------
            soil_wetness = (SM / parameters["FC"][:, j, :]) ** parameters["BETA"][:, j, :]
            soil_wetness = torch.clamp(soil_wetness, min=0.0, max=1.0)
            recharge = (liquid_p[:, j, :] + tosoil) * soil_wetness

            SM = SM + liquid_p[:, j, :] + tosoil - recharge
            excess = SM - parameters["FC"][:, j, :]
            excess = torch.clamp(excess, min=0.0)
            SM = SM - excess
            if "BETAET" in parameters:
                evapfactor = (SM / (parameters["LP"][:, j, :] * parameters["FC"][:, j, :])) ** parameters["BETAET"][
                    :, j, :
                ]
            else:
                evapfactor = SM / (parameters["LP"][:, j, :] * parameters["FC"][:, j, :])
            evapfactor = torch.clamp(evapfactor, min=0.0, max=1.0)
            ETact = et[:, j, :] * evapfactor
            ETact = torch.min(SM, ETact)
            SM = torch.clamp(SM - ETact, min=1e-5)  # SM can not be zero for gradient tracking

            # Groundwater boxes -------------------------------------------------------------------------------------
            SUZ = SUZ + recharge + excess
            PERC = torch.min(SUZ, parameters["PERC"][:, j, :])
            SUZ = SUZ - PERC
            Q0 = parameters["K0"][:, j, :] * torch.clamp(SUZ - parameters["UZL"][:, j, :], min=0.0)
            SUZ = SUZ - Q0
            Q1 = parameters["K1"][:, j, :] * SUZ
            SUZ = SUZ - Q1
            SLZ = SLZ + PERC
            Q2 = parameters["K2"][:, j, :] * SLZ
            SLZ = SLZ - Q2

            # Store time evolution of the internal states
            states["SNOWPACK"][:, j, :] = SNOWPACK
            states["MELTWATER"][:, j, :] = MELTWATER
            states["SM"][:, j, :] = SM
            states["SUZ"][:, j, :] = SUZ
            states["SLZ"][:, j, :] = SLZ

            # total outflow
            out[:, j, 0] = torch.mean(Q0 + Q1 + Q2, dim=1)  # [mm]

        # last states
        final_states = self._get_final_states(states=states)

        return {"y_hat": out, "parameters": parameters, "internal_states": states, "final_states": final_states}

    @property
    def _initial_states(self) -> Dict[str, float]:
        return {"SNOWPACK": 0.001, "MELTWATER": 0.001, "SM": 0.001, "SUZ": 0.001, "SLZ": 0.001}

    @property
    def parameter_ranges(self) -> Dict[str, Tuple[float, float]]:
        return {
            "BETA": (1.0, 6.0),
            "FC": (50.0, 1000.0),
            "K0": (0.05, 0.9),
            "K1": (0.01, 0.5),
            "K2": (0.001, 0.2),
            "LP": (0.2, 1.0),
            "PERC": (0.0, 10.0),
            "UZL": (0.0, 100.0),
            "TT": (-2.5, 2.5),
            "CFMAX": (0.5, 10.0),
            "CFR": (0.0, 0.1),
            "CWH": (0.0, 0.2),
            "BETAET": (0.3, 5.0),
        }


class BasePBM(nn.Module):
    pass


class HBVPBM(BasePBM):
    pass