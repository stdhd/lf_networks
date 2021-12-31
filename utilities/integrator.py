import torch


class Integrator(torch.nn.Module):
    """HGN integrator class: Implements different integration methods for Hamiltonian differential equations.
    """
    METHODS = ["Euler", "RK4", "Leapfrog", "Yoshida", "GatedLeapfrog", "GRU"]

    def __init__(self, delta_t, method="Euler", learn_delta_t=False, do_not_integrate_but_let_unet_sample=False):
        super().__init__()
        self.do_not_integrate_but_let_unet_sample = do_not_integrate_but_let_unet_sample
        """Initialize HGN integrator.

        Args:
            delta_t (float): Time difference between integration steps.
            method (str, optional): Integration method, must be "Euler", "RK4", "Leapfrog" or "Yoshida". Defaults to "Euler".

        Raises:
            KeyError: If the integration method passed is invalid.
        """
        if method not in self.METHODS:
            msg = "%s is not a supported method. " % (method)
            msg += "Available methods are: " + "".join("%s " % m
                                                       for m in self.METHODS)
            raise KeyError(msg)
        if method == 'GatedLeapfrog':
            self.tanh = torch.nn.Tanh()
        if learn_delta_t:
            self.delta_t = torch.nn.Parameter(torch.tensor(delta_t))
        else:
            self.delta_t = delta_t
        self.method = method

    def _get_grads(self, q, p, hnn, remember_energy=False):
        """Apply the Hamiltonian equations to the Hamiltonian network to get dq_dt, dp_dt.

        Args:
            q (torch.Tensor): Latent-space position tensor.
            p (torch.Tensor): Latent-space momentum tensor.
            hnn (HamiltonianNet): Hamiltonian Neural Network.
            remember_energy (bool): Whether to store the computed energy in self.energy.

        Returns:
            tuple(torch.Tensor, torch.Tensor): Position and momentum time derivatives: dq_dt, dp_dt.
        """
        # Compute energy of the system
        energy = hnn(q=q, p=p)

        # dq_dt = dH/dp
        dq_dt = torch.autograd.grad(energy,
                                    p,
                                    create_graph=True,
                                    retain_graph=True,
                                    grad_outputs=torch.ones_like(energy))[0]

        # dp_dt = -dH/dq
        dp_dt = -torch.autograd.grad(energy,
                                     q,
                                     create_graph=True,
                                     retain_graph=True,
                                     grad_outputs=torch.ones_like(energy))[0]

        if remember_energy:
            self.energy = energy.detach().cpu().numpy()

        return dq_dt, dp_dt, energy

    def _euler_step(self, q, p, hnn):
        """Compute next latent-space position and momentum using Euler integration method.

        Args:
            q (torch.Tensor): Latent-space position tensor.
            p (torch.Tensor): Latent-space momentum tensor.
            hnn (HamiltonianNet): Hamiltonian Neural Network.

        Returns:
            tuple(torch.Tensor, torch.Tensor): Next time-step position, momentum and energy: q_next, p_next.
        """
        dq_dt, dp_dt = self._get_grads(q, p, hnn, remember_energy=True)

        # Euler integration
        q_next = q + self.delta_t * dq_dt
        p_next = p + self.delta_t * dp_dt
        return q_next, p_next

    def _euler_step_non_scalar(self, q, p, hnn):
        """Compute next latent-space position and momentum using Euler integration method.

        Args:
            q (torch.Tensor): Latent-space position tensor.
            p (torch.Tensor): Latent-space momentum tensor.
            hnn (HamiltonianNet): Hamiltonian Neural Network.

        Returns:
            tuple(torch.Tensor, torch.Tensor): Next time-step position, momentum and energy: q_next, p_next.
        """
        dq_dt, dp_dt = hnn(q, p)

        # Euler integration
        q_next = q + self.delta_t * dq_dt
        p_next = p + self.delta_t * dp_dt
        return q_next, p_next

    def _rk_step(self, q, p, hnn):
        """Compute next latent-space position and momentum using Runge-Kutta 4 integration method.

        Args:
            q (torch.Tensor): Latent-space position tensor.
            p (torch.Tensor): Latent-space momentum tensor.
            hnn (HamiltonianNet): Hamiltonian Neural Network.

        Returns:
            tuple(torch.Tensor, torch.Tensor): Next time-step position and momentum: q_next, p_next.
        """
        # k1
        k1_q, k1_p = self._get_grads(q, p, hnn, remember_energy=True)

        # k2
        q_2 = q + self.delta_t * k1_q / 2  # x = x_t + dt * k1 / 2
        p_2 = p + self.delta_t * k1_p / 2  # x = x_t + dt * k1 / 2
        k2_q, k2_p = self._get_grads(q_2, p_2, hnn)

        # k3
        q_3 = q + self.delta_t * k2_q / 2  # x = x_t + dt * k2 / 2
        p_3 = p + self.delta_t * k2_p / 2  # x = x_t + dt * k2 / 2
        k3_q, k3_p = self._get_grads(q_3, p_3, hnn)

        # k4
        q_3 = q + self.delta_t * k3_q / 2  # x = x_t + dt * k3
        p_3 = p + self.delta_t * k3_p / 2  # x = x_t + dt * k3
        k4_q, k4_p = self._get_grads(q_3, p_3, hnn)

        # Runge-Kutta 4 integration
        q_next = q + self.delta_t * ((k1_q / 6) + (k2_q / 3) + (k3_q / 3) +
                                     (k4_q / 6))
        p_next = p + self.delta_t * ((k1_p / 6) + (k2_p / 3) + (k3_p / 3) +
                                     (k4_p / 6))
        return q_next, p_next

    def _lf_step(self, q, p, hnn, delta_t=1.):
        """Compute next latent-space position and momentum using LeapFrog integration method.

        Args:
            q (torch.Tensor): Latent-space position tensor.
            p (torch.Tensor): Latent-space momentum tensor.
            hnn (HamiltonianNet): Hamiltonian Neural Network.

        Returns:
            tuple(torch.Tensor, torch.Tensor): Next time-step position and momentum: q_next, p_next.
        """
        # get acceleration
        _, dp_dt, energy = self._get_grads(q, p, hnn, remember_energy=True)
        # leapfrog step
        p_next_half = p + dp_dt * delta_t / 2 # nächste Geschwindigkeit zum zeitpunk t+1/2
        q_next = q + p_next_half * delta_t
        # momentum synchronization
        _, dp_next_dt, energy = self._get_grads(q_next, p_next_half, hnn)
        p_next = p_next_half + dp_next_dt * (delta_t) / 2
        return q_next, p_next, energy.detach().cpu().numpy()

    def _ys_step(self, q, p, hnn):
        """Compute next latent-space position and momentum using 4th order Yoshida integration method.

        Args:
            q (torch.Tensor): Latent-space position tensor.
            p (torch.Tensor): Latent-space momentum tensor.
            hnn (HamiltonianNet): Hamiltonian Neural Network.

        Returns:
            tuple(torch.Tensor, torch.Tensor): Next time-step position and momentum: q_next, p_next.
        """
        # yoshida coeficients c_n and d_m
        w_1 = 1./(2 - 2**(1./3))
        w_0 = -(2**(1./3))*w_1
        c_1 = c_4 = w_1/2.
        c_2 = c_3 = (w_0 + w_1)/2.
        d_1 = d_3 = w_1
        d_2 = w_0

        # first order
        q_1 = q + c_1*p*self.delta_t
        _, a_1 = self._get_grads(q_1, p, hnn, remember_energy=True)
        p_1 = p + d_1*a_1*self.delta_t
        # second order
        q_2 = q_1 + c_2*p_1*self.delta_t
        _, a_2 = self._get_grads(q_2, p, hnn, remember_energy=False)
        p_2 = p_1 + d_2*a_2*self.delta_t
        # third order
        q_3 = q_2 + c_3*p_2*self.delta_t
        _, a_3 = self._get_grads(q_3, p, hnn, remember_energy=False)
        p_3 = p_2 + d_3*a_3*self.delta_t
        # fourth order
        q_4 = q_3 + c_4*p_3*self.delta_t

        return q_4, p_3

    def _lf_step_non_scalar(self, q, p, hnn, delta_t=1.):
        if self.do_not_integrate_but_let_unet_sample:
            q, p = hnn(q, p)
            return q, p
        else:
            delta_t=0.5
            #print('## ', p.size(), q.size()) # torch.Size([4, 64, 4, 4]) torch.Size([4, 64, 4, 4])
            # leapfrog step
            # dq/dt = dH/dp
            # dp/dt = - dH/dq

            #p_next_half = p + dH_dp * self.delta_t / 2
            #q_next = q + p_next_half * self.delta_t
            #p_next = p_next_half - dH_dq * self.delta_t / 2

            # get acceleration
            _, dp_dt = hnn(q, p)
            # leapfrog step
            p_next_half = p + dp_dt * delta_t / 2
            q_next = q + p_next_half * delta_t
            # momentum synchronization
            _, dp_next_dt = hnn(q_next, p_next_half)
            p_next = p_next_half + dp_next_dt * delta_t / 2
            #print('++ ', p_next.size(), p_next.size())
        return q_next, p_next

    def _lf_step_non_scalar_gated(self, q, p, hnn, delta_t=1.):
        # get acceleration
        _, dp_dt = hnn(q, p)
        # leapfrog step
        p_next_half = self.tanh(p + dp_dt * delta_t / 2)
        q_next = self.tanh(q + p_next_half * delta_t)
        # momentum synchronization
        _, dp_next_dt = hnn(q_next, p_next_half)
        p_next = self.tanh(p_next_half + dp_next_dt * delta_t / 2)

        return q_next, p_next

    def step_non_scalar(self, q, p, hnn, context=None):
        if self.method == "Euler":
            return self._euler_step_non_scalar(q, p, hnn)
        if self.method == "Leapfrog":
            return self._lf_step_non_scalar(q, p, hnn, delta_t=self.delta_t)
        if self.method == "GatedLeapfrog":
            return self._lf_step_non_scalar_gated(q, p, hnn, delta_t=self.delta_t)
        raise NotImplementedError

    def step(self, q, p, hnn):
        """Compute next latent-space position and momentum.

        Args:
            q (torch.Tensor): Latent-space position tensor.
            p (torch.Tensor): Latent-space momentum tensor.
            hnn (HamiltonianNet): Hamiltonian Neural Network.

        Raises:
            NotImplementedError: If the integration method requested is not implemented.

        Returns:
            tuple(torch.Tensor, torch.Tensor): Next time-step position and momentum: q_next, p_next.
        """
        if self.method == "Euler":
            return self._euler_step(q, p, hnn)
        if self.method == "RK4":
            return self._rk_step(q, p, hnn)
        if self.method == "Leapfrog":
            return self._lf_step(q, p, hnn)
        if self.method == "Yoshida":
            return self._ys_step(q, p, hnn)
        raise NotImplementedError
