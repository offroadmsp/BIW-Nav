#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CUDA-enabled Tolman-Eichenbaum Machine
Modified to support GPU computation
"""
import numpy as np
import torch
import pdb
import copy
from scipy.stats import truncnorm
import utils

class Model(torch.nn.Module):
    def __init__(self, params, device='cuda'):
        super(Model, self).__init__()
        # 设置设备
        if device == 'cuda':
            self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        elif device == 'cpu':
            self.device = torch.device('cpu')
            # print('finished set cpu')
        print(f"Using device: {self.device}")
        
        self.hyper = copy.deepcopy(params)
        # print(self.hyper.device)
        self.init_trainable()
        # 将整个模型移到指定设备
        self.to(self.device)
    
    def forward(self, walk, prev_iter = None, prev_M = None):
        steps = self.init_walks(prev_iter)
        for g, x, a in walk:
            # 将输入数据移到GPU
            x = x.to(self.device)
            if g is not None:
                g = g.to(self.device) if torch.is_tensor(g) else g
            
            if steps is None:
                steps = [self.init_iteration(g, x, [None for _ in range(len(a))], prev_M)]
            
            L, M, g_gen, p_gen, x_gen, x_logits, x_inf, g_inf, p_inf = self.iteration(x, g, steps[-1].a, steps[-1].M, steps[-1].x_inf, steps[-1].g_inf)
            steps.append(Iteration(g, x, a, L, M, g_gen, p_gen, x_gen, x_logits, x_inf, g_inf, p_inf))    
        
        steps = steps[1:]
        return steps

    def iteration(self, x, locations, a_prev, M_prev, x_prev, g_prev):
        gt_gen, gt_inf = self.gen_g(a_prev, g_prev, locations)        
        x_inf, g_inf, p_inf_x, p_inf = self.inference(x, locations, M_prev, x_prev, gt_inf)                        
        x_gen, x_logits, p_gen = self.generative(M_prev, p_inf, g_inf, gt_gen)
        
        M = [self.hebbian(M_prev[0], torch.cat(p_inf,dim=1), torch.cat(p_gen,dim=1))]
        
        if self.hyper['use_p_inf']:
            M.append(M[0] if self.hyper['common_memory'] else self.hebbian(M_prev[1], torch.cat(p_inf,dim=1), torch.cat(p_inf_x,dim=1), do_hierarchical_connections=False))
        
        L = self.loss(gt_gen, p_gen, x_logits, x, g_inf, p_inf, p_inf_x, M_prev)
        return L, M, gt_gen, p_gen, x_gen, x_logits, x_inf, g_inf, p_inf
        
    def inference(self, x, locations, M_prev, x_prev, g_gen):
        x_c = self.f_c(x)
        x_f = self.x_prev2x(x_prev, x_c)
        x_ = self.x2x_(x_f)
        p_x = self.attractor(x_, M_prev[1], retrieve_it_mask=self.hyper['p_retrieve_mask_inf']) if self.hyper['use_p_inf'] else None
        g = self.inf_g(p_x, g_gen, x, locations)
        g_ = self.g2g_(g)
        p = self.inf_p(x_, g_)
        return x_f, g, p_x, p    

    def generative(self, M_prev, p_inf, g_inf, g_gen):
        x_p, x_p_logits = self.gen_x(p_inf[0])
        p_g_inf = self.gen_p(g_inf, M_prev[0])
        x_g, x_g_logits = self.gen_x(p_g_inf[0])
        p_g_gen = self.gen_p(g_gen, M_prev[0])
        x_gt, x_gt_logits = self.gen_x(p_g_gen[0])
        return (x_p, x_g, x_gt), (x_p_logits, x_g_logits, x_gt_logits), p_g_inf

    def loss(self, g_gen, p_gen, x_logits, x, g_inf, p_inf, p_inf_x, M_prev):
        L_p_g = torch.sum(torch.stack(utils.squared_error(p_inf, p_gen), dim=0), dim=0)
        L_p_x = torch.sum(torch.stack(utils.squared_error(p_inf, p_inf_x), dim=0), dim=0) if self.hyper['use_p_inf'] else torch.zeros_like(L_p_g)
        L_g = torch.sum(torch.stack(utils.squared_error(g_inf, g_gen), dim=0), dim=0)         
        labels = torch.argmax(x, 1)            
        L_x_gen = utils.cross_entropy(x_logits[2], labels)
        L_x_g = utils.cross_entropy(x_logits[1], labels)
        L_x_p = utils.cross_entropy(x_logits[0], labels)
        L_reg_g = torch.sum(torch.stack([torch.sum(g ** 2, dim=1) for g in g_inf], dim=0), dim=0)
        L_reg_p = torch.sum(torch.stack([torch.sum(torch.abs(p), dim=1) for p in p_inf], dim=0), dim=0)
        L = [L_p_g, L_p_x, L_x_gen, L_x_g, L_x_p, L_g, L_reg_g, L_reg_p]
        return L

    def init_trainable(self):
        self.alpha = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(np.log(self.hyper['f_initial'][f] / (1 - self.hyper['f_initial'][f])), dtype=torch.float)) for f in range(self.hyper['n_f'])])
        self.w_x = torch.nn.Parameter(torch.tensor(1.0))
        self.b_x = torch.nn.Parameter(torch.zeros(self.hyper['n_x_c']))
        self.w_p = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(1.0)) for f in range(self.hyper['n_f'])])        
        self.g_init = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(truncnorm.rvs(-2, 2, size=self.hyper['n_g'][f], loc=0, scale=self.hyper['g_init_std']), dtype=torch.float)) for f in range(self.hyper['n_f'])])
        self.logsig_g_init = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(truncnorm.rvs(-2, 2, size=self.hyper['n_g'][f], loc=0, scale=self.hyper['g_init_std']), dtype=torch.float)) for f in range(self.hyper['n_f'])])                
        self.MLP_D_a = MLP([self.hyper['n_actions'] for _ in range(self.hyper['n_f'])],
                            [sum([self.hyper['n_g'][f_from] for f_from in range(self.hyper['n_f']) if self.hyper['g_connections'][f_to][f_from]])*self.hyper['n_g'][f_to] for f_to in range(self.hyper['n_f'])],
                            activation=[torch.tanh, None],
                            hidden_dim=[self.hyper['d_hidden_dim'] for _ in range(self.hyper['n_f'])],
                            bias=[True, False])        
        self.MLP_D_a.set_weights(1, 0.0)
        self.D_no_a = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(sum([self.hyper['n_g'][f_from] for f_from in range(self.hyper['n_f']) if self.hyper['g_connections'][f_to][f_from]])*self.hyper['n_g'][f_to])) for f_to in range(self.hyper['n_f'])])
        self.MLP_sigma_g_path = MLP(self.hyper['n_g'], self.hyper['n_g'], activation=[torch.tanh, torch.exp], hidden_dim=[2 * g for g in self.hyper['n_g']])
        self.MLP_sigma_p = MLP(self.hyper['n_p'], self.hyper['n_p'], activation=[torch.tanh, torch.exp])
        self.MLP_mu_g_mem = MLP(self.hyper['n_g_subsampled'], self.hyper['n_g'], hidden_dim=[2 * g for g in self.hyper['n_g']])
        self.MLP_mu_g_mem.set_weights(-1, [torch.tensor(truncnorm.rvs(-2, 2, size=list(self.MLP_mu_g_mem.w[f][-1].weight.shape), loc=0, scale=self.hyper['g_mem_std']), dtype=torch.float) for f in range(self.hyper['n_f'])])
        self.MLP_sigma_g_mem = MLP([2 for _ in self.hyper['n_g_subsampled']], self.hyper['n_g'], activation=[torch.tanh, torch.exp], hidden_dim=[2 * g for g in self.hyper['n_g']])
        self.MLP_mu_g_shiny = MLP([1 for _ in range(self.hyper['n_f_ovc'] if self.hyper['separate_ovc'] else self.hyper['n_f'])], 
                                  [n_g for n_g in self.hyper['n_g'][(self.hyper['n_f_g'] if self.hyper['separate_ovc'] else 0):]], 
                                  hidden_dim=[2*n_g for n_g in self.hyper['n_g'][(self.hyper['n_f_g'] if self.hyper['separate_ovc'] else 0):]])
        self.MLP_sigma_g_shiny = MLP([1 for _ in range(self.hyper['n_f_ovc'] if self.hyper['separate_ovc'] else self.hyper['n_f'])],
                                     [n_g for n_g in self.hyper['n_g'][(self.hyper['n_f_g'] if self.hyper['separate_ovc'] else 0):]],
                                     hidden_dim=[2*n_g for n_g in self.hyper['n_g'][(self.hyper['n_f_g'] if self.hyper['separate_ovc'] else 0):]], activation=[torch.tanh, torch.exp])
        self.MLP_c_star = MLP(self.hyper['n_x_f'][0], self.hyper['n_x'], hidden_dim=20 * self.hyper['n_x_c'])
    
    def init_iteration(self, g, x, a, M):
        self.hyper['batch_size'] = x.shape[0]
        if M is None:
            M = [torch.zeros((self.hyper['batch_size'],sum(self.hyper['n_p']),sum(self.hyper['n_p'])), dtype=torch.float, device=self.device)]
            if self.hyper['use_p_inf']:
                M.append(M[0] if self.hyper['common_memory'] else torch.zeros((self.hyper['batch_size'],sum(self.hyper['n_p']),sum(self.hyper['n_p'])), dtype=torch.float, device=self.device)) 
        
        g_inf = [torch.stack([self.g_init[f] for _ in range(self.hyper['batch_size'])]) for f in range(self.hyper['n_f'])]
        x_inf = [torch.zeros((self.hyper['batch_size'], self.hyper['n_x_f'][f]), device=self.device) for f in range(self.hyper['n_f'])]        
        return Iteration(g=g, x=x, a=a, M=M, x_inf=x_inf, g_inf=g_inf)    
    
    def init_walks(self, prev_iter):
        if prev_iter is not None:   
            for a_i, a in enumerate(prev_iter[0].a):
                if a is None:
                    for M in prev_iter[0].M:
                        M[a_i,:,:] = 0         
                    for f, g_inf in enumerate(prev_iter[0].g_inf):
                        g_inf[a_i,:] = self.g_init[f]
                    for f, x_inf in enumerate(prev_iter[0].x_inf):
                        x_inf[a_i,:] = torch.zeros(self.hyper['n_x_f'][f], device=self.device)
        return prev_iter
    
    # 其他方法保持不变，但需要确保在生成随机数时使用正确的设备
    def gen_g(self, a_prev, g_prev, locations):
        mu_g = self.f_mu_g_path(a_prev, g_prev)
        sigma_g = self.f_sigma_g_path(a_prev, g_prev)
        # 使用torch.randn并指定设备
        g = [mu_g[f] + sigma_g[f] * torch.randn_like(sigma_g[f]) if self.hyper['do_sample'] else mu_g[f] for f in range(self.hyper['n_f'])]
        shiny_envs = [location['shiny'] is not None for location in locations]
        g_gen = self.f_mu_g_path(a_prev, g_prev, no_direc=shiny_envs) if any(shiny_envs) else g
        return g_gen, (g, sigma_g)
    
    def gen_p(self, g, M_prev):
        g_ = self.g2g_(g)
        mu_p = self.attractor(g_, M_prev, retrieve_it_mask=self.hyper['p_retrieve_mask_gen'])
        sigma_p = self.f_sigma_p(mu_p)
        # 使用torch.randn_like确保在正确的设备上
        p = [mu_p[f] + sigma_p[f] * torch.randn_like(sigma_p[f]) if self.hyper['do_sample'] else mu_p[f] for f in range(self.hyper['n_f'])]
        return p

    # 继续其他方法...
    def gen_x(self, p):
        if self.hyper['do_sample']:
            x, logits = self.f_x(p)
        else:
            x, logits = self.f_x(p)
        return x, logits
        
    def inf_g(self, p_x, g_gen, x, locations):
        if self.hyper['use_p_inf']:
            g_downsampled = [torch.matmul(p_x[f], torch.t(self.hyper['W_repeat'][f])) for f in range(self.hyper['n_f'])]      
            mu_g_mem = self.f_mu_g_mem(g_downsampled)
            with torch.no_grad():
                x_hat, x_hat_logits = self.gen_x(p_x[0])            
                err = utils.squared_error(x, x_hat)
            sigma_g_input = [torch.cat((torch.sum(g ** 2, dim=1, keepdim=True), torch.unsqueeze(err, dim=1)), dim=1) for g in mu_g_mem]            
            mu_g_mem = self.f_g_clamp(mu_g_mem)
            sigma_g_mem = self.f_sigma_g_mem(sigma_g_input)        
        
        mu_g_path = g_gen[0]
        sigma_g_path = g_gen[1]
        mu_g, sigma_g = [], []
        for f in range(self.hyper['n_f']):
            if self.hyper['use_p_inf']:
                mu, sigma = utils.inv_var_weight([mu_g_path[f], mu_g_mem[f]],[sigma_g_path[f], sigma_g_mem[f]])
            else:
                mu, sigma = mu_g_path[f], sigma_g_path[f]
            mu_g.append(mu)
            sigma_g.append(sigma)
        
        shiny_envs = [location['shiny'] is not None for location in locations]
        if any(shiny_envs):            
            shiny_locations = torch.unsqueeze(torch.stack([torch.tensor(location['shiny'], dtype=torch.float, device=self.device) for location in locations if location['shiny'] is not None]), dim=-1)
            mu_g_shiny = self.f_mu_g_shiny([shiny_locations for _ in range(self.hyper['n_f_g'] if self.hyper['separate_ovc'] else self.hyper['n_f'])])
            sigma_g_shiny = self.f_sigma_g_shiny([shiny_locations for _ in range(self.hyper['n_f_g'] if self.hyper['separate_ovc'] else self.hyper['n_f'])])
            module_start = self.hyper['n_f_g'] if self.hyper['separate_ovc'] else 0
            for f in range(module_start, self.hyper['n_f']):
                mu, sigma = utils.inv_var_weight([mu_g[f][shiny_envs,:], mu_g_shiny[f - module_start]], [sigma_g[f][shiny_envs,:], sigma_g_shiny[f - module_start]])                
                mask = torch.zeros_like(mu_g[f], dtype=torch.bool)
                mask[shiny_envs,:] = True
                mu_g[f] = mu_g[f].masked_scatter(mask,mu) 
                sigma_g[f] = sigma_g[f].masked_scatter(mask,sigma) 
        
        # 使用torch.randn_like
        g = [mu_g[f] + sigma_g[f] * torch.randn_like(sigma_g[f]) if self.hyper['do_sample'] else mu_g[f] for f in range(self.hyper['n_f'])]
        return g
        
    def inf_p(self, x_, g_):
        p = []
        for f in range(self.hyper['n_f']):
            mu_p = self.f_p(g_[f] * x_[f])
            sigma_p = 0
            if self.hyper['do_sample']:
                p.append(mu_p + sigma_p * torch.randn_like(mu_p))
            else:
                p.append(mu_p)
        return p

    # 剩余方法保持不变...
    def x_prev2x(self, x_prev, x_c):
        alpha = [torch.nn.Sigmoid()(self.alpha[f]) for f in range(self.hyper['n_f'])]
        x = [(1 - alpha[f]) * x_prev[f] + alpha[f] * x_c for f in range(self.hyper['n_f'])]
        return x
    
    def x2x_(self, x):
        normalised = self.f_n(x)
        x_ = [torch.nn.Sigmoid()(self.w_p[f]) * torch.matmul(normalised[f],self.hyper['W_tile'][f]) for f in range(self.hyper['n_f'])]        
        return x_

    def g2g_(self, g):
        downsampled = self.f_g(g)
        g_ = [torch.matmul(downsampled[f], self.hyper['W_repeat'][f]) for f in range(self.hyper['n_f'])]
        return g_            
        
    def f_mu_g_path(self, a_prev, g_prev, no_direc=None):
        no_direc = [False for _ in a_prev] if no_direc is None else no_direc
        a_prev_step = [a if a is not None else 0 for a in a_prev]
        a_do_step = [a != None for a in a_prev]        
        
        if self.hyper['has_static_action']:
            a = torch.zeros((len(a_prev_step),self.hyper['n_actions']), device=self.device).scatter_(1, torch.clamp(torch.tensor(a_prev_step, device=self.device).unsqueeze(1)-1,min=0), 1.0*(torch.tensor(a_prev_step, device=self.device).unsqueeze(1)>0))
        else:
            a = torch.zeros((len(a_prev_step),self.hyper['n_actions']), device=self.device).scatter_(1, torch.tensor(a_prev_step, device=self.device).unsqueeze(1), 1.0)
        
        D_a = self.MLP_D_a([a for _ in range(self.hyper['n_f'])])
        for f in range(self.hyper['n_f']):
            D_a[f][no_direc,:] = self.D_no_a[f]
        
        D_a = [torch.reshape(D_a[f_to],(-1, sum([self.hyper['n_g'][f_from] for f_from in range(self.hyper['n_f']) if self.hyper['g_connections'][f_to][f_from]]), self.hyper['n_g'][f_to])) for f_to in range(self.hyper['n_f'])]        
        g_in = [torch.unsqueeze(torch.cat([g_prev[f_from] for f_from in range(self.hyper['n_f']) if self.hyper['g_connections'][f_to][f_from]], dim=1),1) for f_to in range(self.hyper['n_f'])]        
        delta = [torch.squeeze(torch.matmul(g, T)) for g, T in zip(g_in, D_a)]
        g_step = [g + d if g.dim() > 1 else torch.unsqueeze(g + d, 0) for g, d in zip(g_prev, delta)]
        g_step = self.f_g_clamp(g_step)
        
        return [torch.stack([g_step[f][batch_i, :] if do_step else self.g_init[f] for batch_i, do_step in enumerate(a_do_step)]) for f in range(self.hyper['n_f'])]
    
    def f_sigma_g_path(self, a_prev, g_prev):
        a_do_step = [a != None for a in a_prev]
        from_g = self.MLP_sigma_g_path(g_prev)
        from_prior = [torch.exp(logsig) for logsig in self.logsig_g_init]
        return [torch.stack([from_g[f][batch_i, :] if do_step else from_prior[f] for batch_i, do_step in enumerate(a_do_step)]) for f in range(self.hyper['n_f'])]

    def f_mu_g_mem(self, g_downsampled):
        return self.MLP_mu_g_mem(g_downsampled)
    
    def f_sigma_g_mem(self, g_downsampled):
        sigma = self.MLP_sigma_g_mem(g_downsampled)
        return [sigma[f] + self.hyper['p2g_scale_offset'] * self.hyper['p2g_sig_val'] for f in range(self.hyper['n_f'])]
    
    def f_mu_g_shiny(self, shiny):
        mu_g = self.MLP_mu_g_shiny(shiny)
        mu_g = [torch.abs(mu) for mu in mu_g]
        g = self.f_p(mu_g)
        return g
        
    def f_sigma_g_shiny(self, shiny):
        return self.MLP_sigma_g_shiny(shiny)    
    
    def f_sigma_p(self, p):
        return self.MLP_sigma_p(p)
    
    def f_x(self, p):
        x = self.w_x * torch.matmul(p, torch.t(self.hyper['W_tile'][0])) + self.b_x
        logits = self.f_c_star(x)
        probability = utils.softmax(logits) 
        return probability, logits
    
    def f_c_star(self, compressed):
        return self.MLP_c_star(compressed)
    
    def f_c(self, decompressed):
        # 需要确保two_hot_table在正确的设备上
        indices = torch.argmax(decompressed, dim=1)
        return torch.stack([self.hyper['two_hot_table'][i.item()] for i in indices], dim=0).to(self.device)
    
    def f_n(self, x):
        normalised = [utils.normalise(utils.relu(x[f] - torch.mean(x[f]))) for f in range(self.hyper['n_f'])]
        return normalised
    
    def f_g(self, g):
        downsampled = [torch.matmul(g[f], self.hyper['g_downsample'][f]) for f in range(self.hyper['n_f'])]
        return downsampled    
    
    def f_g_clamp(self, g):
        activation = [torch.clamp(g_f, min=-1, max=1) for g_f in g]
        return activation    
    
    def f_p(self, p):
        activation = [utils.leaky_relu(torch.clamp(p_f, min=-1, max=1)) for p_f in p] if type(p) is list else utils.leaky_relu(torch.clamp(p, min=-1, max=1)) 
        return activation        
    
    def attractor(self, p_query, M, retrieve_it_mask=None):        
        h_t = torch.cat(p_query, dim=1)
        h_t = self.f_p(h_t)        
        retrieve_it_mask = [torch.ones(sum(self.hyper['n_p']), device=self.device) for _ in range(self.hyper['n_p'])] if retrieve_it_mask is None else retrieve_it_mask
        
        for tau in range(self.hyper['i_attractor']):
            h_t = (1-retrieve_it_mask[tau])*h_t + retrieve_it_mask[tau]*(self.f_p(self.hyper['kappa'] * h_t + torch.squeeze(torch.matmul(torch.unsqueeze(h_t,1), M))))
        
        n_p = np.cumsum(np.concatenate(([0],self.hyper['n_p'])))                
        p = [h_t[:,n_p[f]:n_p[f+1]] for f in range(self.hyper['n_f'])]
        return p
    
    def hebbian(self, M_prev, p_inferred, p_generated, do_hierarchical_connections=True):
        M_new = torch.squeeze(torch.matmul(torch.unsqueeze(p_inferred + p_generated, 2),torch.unsqueeze(p_inferred - p_generated,1)))
        if do_hierarchical_connections:
            M_new = M_new * self.hyper['p_update_mask']
        M = torch.clamp(self.hyper['lambda'] * M_prev + self.hyper['eta'] * M_new, min=-1, max=1)
        return M


# MLP和其他类保持不变
class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, activation=(torch.nn.functional.elu, None), hidden_dim=None, bias=(True, True)):
        super(MLP, self).__init__()        
        if type(in_dim) is list:
            self.is_list = True
        else:
            in_dim = [in_dim]
            out_dim = [out_dim]
            self.is_list = False
        
        self.N = len(in_dim)
        self.w = torch.nn.ModuleList([])
        for n in range(self.N):
            if hidden_dim is None:
                hidden = int(np.mean([in_dim[n],out_dim[n]]))
            else:
                hidden = hidden_dim[n] if self.is_list else hidden_dim         
            self.w.append(torch.nn.ModuleList([torch.nn.Linear(in_dim[n], hidden, bias=bias[0]), torch.nn.Linear(hidden, out_dim[n], bias=bias[1])]))
        
        self.activation = activation
        with torch.no_grad():
            for from_layer in range(2):
                for n in range(self.N):
                    torch.nn.init.xavier_normal_(self.w[n][from_layer].weight)
                    if bias[from_layer]:
                        self.w[n][from_layer].bias.fill_(0.0)        
    
    def set_weights(self, from_layer, value):
        if type(value) is not list:
            input_value = [value for n in range(self.N)]
        else:
            input_value = value
        
        with torch.no_grad():
            for n in range(self.N):
                if type(input_value[n]) is torch.Tensor:
                    self.w[n][from_layer].weight.copy_(input_value[n])                    
                else:
                    self.w[n][from_layer].weight.fill_(input_value[n]) 
        
    def forward(self, data):
        if self.is_list:
            input_data = data
        else:
            input_data = [data]
        
        output = []
        for n in range(self.N):
            module_output = self.w[n][0](input_data[n])
            if self.activation[0] is not None:
                module_output = self.activation[0](module_output)
            module_output = self.w[n][1](module_output)
            if self.activation[1] is not None:
                module_output = self.activation[1](module_output)
            output.append(module_output) 
        
        if not self.is_list:
            output = output[0]
        return output     


class LSTM(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers=1, n_a=4):
        super(LSTM, self).__init__()
        self.lstm = torch.nn.LSTM(in_dim, hidden_dim, n_layers, batch_first=True)
        self.lin = torch.nn.Linear(hidden_dim, out_dim)
        self.n_a = n_a
        
    def forward(self, data, prev_hidden=None):
        if prev_hidden is None:
            hidden_state = torch.randn(self.lstm.num_layers, data.shape[0], self.lstm.hidden_size, device=data.device)
            cell_state = torch.randn(self.lstm.num_layers, data.shape[0], self.lstm.hidden_size, device=data.device)
            prev_hidden = (hidden_state, cell_state)
        
        lstm_out, lstm_hidden = self.lstm(data, prev_hidden)
        lin_out = self.lin(lstm_out)
        out = utils.softmax(lin_out)
        return out, lstm_hidden

    def prepare_data(self, data_in):
        device = data_in[0][1].device if torch.is_tensor(data_in[0][1]) else 'cpu'
        actions = [torch.zeros((len(step[2]),self.n_a), device=device).scatter_(1, torch.tensor(step[2], device=device).unsqueeze(1), 1.0) for step in data_in]
        vectors = [torch.cat((step[1], action), dim=1) for step, action in zip(data_in, actions)]
        data = torch.stack(vectors, dim=1)
        return data  


class Iteration:
    def __init__(self, g=None, x=None, a=None, L=None, M=None, g_gen=None, p_gen=None, x_gen=None, x_logits=None, x_inf=None, g_inf=None, p_inf=None):
        self.g = g
        self.x = x
        self.a = a
        self.L = L
        self.M = M
        self.g_gen = g_gen
        self.p_gen = p_gen
        self.x_gen = x_gen
        self.x_logits = x_logits
        self.x_inf = x_inf
        self.g_inf = g_inf
        self.p_inf = p_inf

    def correct(self):
        observation = self.x.detach().cpu().numpy()
        predictions = [tensor.detach().cpu().numpy() for tensor in self.x_gen]
        accuracy = [np.argmax(prediction, axis=-1) == np.argmax(observation, axis=-1) for prediction in predictions]
        return accuracy
    
    def detach(self):
        self.L = [tensor.detach() for tensor in self.L]
        self.M = [tensor.detach() for tensor in self.M]
        self.g_gen = [tensor.detach() for tensor in self.g_gen]
        self.p_gen = [tensor.detach() for tensor in self.p_gen]
        self.x_gen = [tensor.detach() for tensor in self.x_gen]
        self.x_inf = [tensor.detach() for tensor in self.x_inf]
        self.g_inf = [tensor.detach() for tensor in self.g_inf]
        self.p_inf = [tensor.detach() for tensor in self.p_inf]
        return self