from transformers import PretrainedConfig


class DiffusionGemmaTextConfig(PretrainedConfig):
    model_type = "diffusion_gemma_text"
    base_config_key = "text_config"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class DiffusionGemmaVisionConfig(PretrainedConfig):
    model_type = "diffusion_gemma_vision"
    base_config_key = "vision_config"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class DiffusionGemmaConfig(PretrainedConfig):
    model_type = "diffusion_gemma"
    sub_configs = {
        "text_config": DiffusionGemmaTextConfig,
        "vision_config": DiffusionGemmaVisionConfig,
    }

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        canvas_length=256,
        vision_soft_tokens_per_image=None,
        image_token_id=None,
        boi_token_id=None,
        eoi_token_id=None,
        tie_word_embeddings=True,
        **kwargs,
    ):
        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"]()
        else:
            self.text_config = text_config

        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()
        else:
            self.vision_config = vision_config

        self.canvas_length = canvas_length
        self.vision_soft_tokens_per_image = vision_soft_tokens_per_image
        self.image_token_id = image_token_id
        self.boi_token_id = boi_token_id
        self.eoi_token_id = eoi_token_id
        super().__init__(**kwargs, tie_word_embeddings=tie_word_embeddings)
