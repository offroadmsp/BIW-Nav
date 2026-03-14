#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Program:  cann_base.py
Modified for BIW: Implements effective time constant modulation.
Dynamics: tau_eff(alpha) * du/dt = -u + Rec + Input
"""

import numpy as np

class cann_model:
    def __init__(self, args):
        # 1. 基础网格参数
        self.z_range = [-np.pi, np.pi]
        self.N = getattr(args, 'N', 100)
        self.dx = (self.z_range[1] - self.z_range[0]) / self.N
        self.x = np.linspace(self.z_range[0], self.z_range[1], self.N, endpoint=False)
        
        # 2. 动力学参数 (Key for Fig.7)
        self.tau_0 = getattr(args, 'tau', 10.0)    # 基准时间常数
        self.alpha = getattr(args, 'alpha', 1.0)   # 时间尺度因子 [0, 1]
        self.epsilon = getattr(args, 'eps', 0.1)   # 避免奇异性
        
        # Formula: tau_eff(alpha) = tau_0 * (eps + (1-eps)*alpha)
        # alpha=0 -> Fast (tau ~ eps*tau0)
        # alpha=1 -> Slow (tau ~ tau0)
        self.tau_eff = self.tau_0 * (self.epsilon + (1.0 - self.epsilon) * self.alpha)
        
        self.a = getattr(args, 'a', 0.5)           # 空间交互宽度
        self.k = getattr(args, 'k', 0.1)           # 全局抑制
        self.A = getattr(args, 'A', 1.0)           # 输入增益
        
        # 3. 递归连接矩阵 J (Gaussian Interaction)
        dist = np.abs(self.x[:, None] - self.x[None, :])
        dist = np.minimum(dist, 2*np.pi - dist) # Periodic boundary
        self.J = (1.0 / (np.sqrt(2 * np.pi) * self.a)) * np.exp(-dist**2 / (2 * self.a**2))
        
        # 4. 状态变量
        self.u = np.zeros(self.N)
        self.input = np.zeros(self.N)
        self.rho = self.N / (2 * np.pi)

    def set_input(self, A, z0):
        """设置高斯外部输入"""
        dist = np.abs(self.x - z0)
        dist = np.minimum(dist, 2*np.pi - dist)
        self.input = A * np.exp(-dist**2 / (4 * self.a**2))

    def get_r(self, u):
        """Divisive Normalization"""
        u_rect = np.maximum(u, 0)
        denom = 1.0 + self.k * self.rho * np.sum(u_rect**2) * self.dx
        return u_rect**2 / (denom + 1e-9)

    def get_dudt(self, t, u):
        """
        Dynamics: tau_eff * du/dt = -u + J*r + I
        Output: du/dt
        """
        r = self.get_r(u)
        interaction = np.dot(self.J, r) * self.dx * self.rho
        # 核心：除以 tau_eff
        dudt = (-u + interaction + self.input) / self.tau_eff
        return dudt

    def cm_of_u(self):
        """Population Vector Decoding"""
        r = self.get_r(self.u)
        Z = np.sum(r * np.exp(1j * self.x))
        return np.angle(Z)