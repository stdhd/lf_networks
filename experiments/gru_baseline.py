from experiments.FirstStageExperimentGanLoss import FirstStageTrainerGAN
from networks.e2e_predictor import EndToEndPredictor
from networks.conv_gru import ConvGRU
import torch


class GruBaselineTrainer(FirstStageTrainerGAN):
    def __init__(self, params, *args, **kwargs, ):
        super().__init__(params)

    def set_model(self):
        self.model = GruBaselineModel(self.params)


class GruBaselineModel(EndToEndPredictor):
    def integrate_step(self, q, p, step_no, hidden_state):
        x = torch.cat([q, p], dim=1)
        if step_no == 0:
            #prev_state = [x] * self.params['networks']['gru']['n_layers']
            prev_state = [None] * self.params['networks']['gru']['n_layers']
        else:
            prev_state = hidden_state

        upd_hidden = self.hnn(x, prev_state)
        result = upd_hidden[-1]
        q, p = result[:, :result.size(1) // 2], result[:, result.size(1) // 2:]

        return q, p, None, upd_hidden

    def post_process_rollouts_to_q(self, q, p):
        return q

