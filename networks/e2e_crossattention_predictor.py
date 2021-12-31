from networks.e2e_predictor import EndToEndPredictor
from networks.openai_guided_diffusion.openai_unet import CAUNetModelNoEmbeddings
from networks.startframe_encoder import get_encoder


class CrossAttPredictor(EndToEndPredictor):
    def __init__(self, params, *args, **kwargs,):
        super().__init__(params, *args, **kwargs)
        self.hnn = CAUNetModelNoEmbeddings(in_channels=params["latent_dimensions"]["pq_latent_size"],
                                           out_channels=params["latent_dimensions"]["pq_latent_size"],
                                           **params['networks']['oaiunet'])
        self.state_to_context_encoder = get_encoder(in_channels=64, **self.params["networks"]['context_encoder'])

    def forward(self, rollout_batch=None, conditioning_frame=None, prediction=None, n_steps=4, initial_p=None, p_use_true_q=-1):
        if initial_p is None:
            context, mu, logvar = self.motion_encoder(rollout_batch[:, :])

            context = context.view((context.size(0), self.params['latent_dimensions']['pq_latent_size'],
                                self.params['latent_dimensions']['pq_latent_dim'],
                                self.params['latent_dimensions']['pq_latent_dim']))

            prediction_shape = list(rollout_batch.shape)
        else:
            raise NotImplementedError

        prediction_shape[1] = n_steps
        previously_predicted_frame = conditioning_frame
        recurrent_state = self.start_encoder(previously_predicted_frame)
        for i in range(n_steps):
            recurrent_state = self.hnn(recurrent_state, None, context=context)
            context = self.state_to_context_encoder(recurrent_state)
            if self.params['optimization'].get('spade_frame_equals_previous', False):
                spade_frame = rollout_batch[:, i - 1]
            else:
                spade_frame = conditioning_frame
            x_reconstructed = self.decoder(recurrent_state, spade_frame)
            prediction.append_reconstruction(x_reconstructed)

        return prediction, 0

    def sample(self, p, conditioning_frame, n_steps, prediction):
        previously_predicted_frame = conditioning_frame
        recurrent_state = self.start_encoder(previously_predicted_frame)
        context = p
        for i in range(n_steps):
            recurrent_state = self.hnn(recurrent_state, None, context=context)
            context = self.state_to_context_encoder(recurrent_state)
            spade_frame = conditioning_frame
            x_reconstructed = self.decoder(recurrent_state, spade_frame)
            prediction.append_reconstruction(x_reconstructed)
        return prediction







