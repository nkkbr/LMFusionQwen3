from transformers import Qwen3Config

class LMFusionQwen3Config(Qwen3Config):
    model_type = 'lmfusion_qwen3'

    def __init__(
        self,
        vision_config=None,
        boi_token_id=None,
        eoi_token_id=None,
        vae_model_name_or_path="stabilityai/sd-vae-ft-mse",
        loss_lambda=5.,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # This line is necessary, as many settings are configured in the __init__ code of Qwen3Config.
        # Although these parameters, in our initial weight file, would be received by **kwargs even if we didn't
        # call super().__init__(**kwargs), self.layer_types is generated within __init__.
        # It is necessary for us to ensure it gets executed.

        if vision_config is None:
            vision_config = {
                "in_channels":4,
                "out_channels":4,
                "temb_channels":1280,
                "hidden_size": kwargs.get("hidden_size", 4096)
            }
        
        self.vision_config = vision_config
        self.boi_token_id = boi_token_id
        self.eoi_token_id = eoi_token_id
        self.vae_model_name_or_path = vae_model_name_or_path
        self.loss_lambda = loss_lambda

        # If there are more new parameters, we need to continue adding them...