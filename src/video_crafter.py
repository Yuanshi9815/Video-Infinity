import torch

from diffusers.models import AutoencoderKL, UNet3DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.schedulers import DPMSolverMultistepScheduler
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_synth import TextToVideoSDPipeline
from diffusers.configuration_utils import register_to_config
from diffusers.models.embeddings import TimestepEmbedding, Timesteps

class VideoCrafterPipeline(TextToVideoSDPipeline):
    @register_to_config
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet3DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        fps_cond: bool = True,
    ):
        self.fps_cond = fps_cond
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler, 
        )

    @torch.no_grad()
    def __call__(
        self,
        *args,
        **kwargs,
    ):
        fixed_fps = kwargs.pop("fps", 24)
        def post_function(sample):
            fps = fixed_fps
            unet = self.unet
            if self.fps_cond:
                fps = torch.tensor([fps], dtype=torch.float64 , device=sample.device)
                fps_emb = unet.fps_proj(fps)
                fps_emb = fps_emb.to(sample.device, dtype=unet.dtype)
                fps_emb = unet.fps_embedding(fps_emb).repeat_interleave(repeats=sample.shape[0], dim=0)
                sample += fps_emb
            return sample
        self.unet.time_embedding.post_act = post_function
        # kwargs.pop("fps", None)
        return super().__call__(*args, **kwargs)
        
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs,
    ):
        pipe = TextToVideoSDPipeline.from_pretrained("cerspense/zeroscope_v2_576w", **kwargs)
        pipe.__class__ = cls
        pipe.fps_cond = True
        pipe.unet = UNetVideoCrafter.from_pretrained(pretrained_model_name_or_path, **kwargs)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras=True, algorithm_type="sde-dpmsolver++")
        return pipe

class UNetVideoCrafter(UNet3DConditionModel):
    @register_to_config
    def __init__(
        self,
        sample_size,
        in_channels,
        out_channels,
        down_block_types,
        up_block_types,
        block_out_channels,
        layers_per_block,
        downsample_padding,
        mid_block_scale_factor,
        act_fn,
        norm_num_groups,
        norm_eps,
        cross_attention_dim,
        attention_head_dim,
        num_attention_heads,
        fps_cond: bool = True,
        **kwargs
    ):
        self.fps_cond = fps_cond

        super().__init__(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            downsample_padding=downsample_padding,
            mid_block_scale_factor=mid_block_scale_factor,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            cross_attention_dim=cross_attention_dim,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            **kwargs
        )

        if self.fps_cond:
            self.fps_proj = Timesteps(block_out_channels[0], True, 0)
            self.fps_embedding = TimestepEmbedding(
                    block_out_channels[0],
                    block_out_channels[0] * 4,
                    act_fn=act_fn,
                )
