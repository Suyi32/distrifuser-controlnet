import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, UNet2DConditionModel, StableDiffusionXLPipelineControlNet

from .models.distri_sdxl_unet_pp import DistriUNetPP
from .models.distri_sdxl_unet_tp import DistriUNetTP
from .models.naive_patch_sdxl import NaivePatchUNet
from .models.distri_sdxl_controlnet_pp import DistriControlnetPP
from .utils import DistriConfig, PatchParallelismCommManager

from copy import deepcopy

class DistriSDXLPipeline:
    def __init__(self, pipeline: StableDiffusionXLPipeline, module_config: DistriConfig):
        self.pipeline = pipeline
        self.distri_config = module_config

        self.static_inputs = None

        self.prepare()
        print("DistriSDXLPipeline initialized. Prepare done.")

    @staticmethod
    def from_pretrained(distri_config: DistriConfig, **kwargs):
        device = distri_config.device
        pretrained_model_name_or_path = kwargs.pop(
            "pretrained_model_name_or_path", "stabilityai/stable-diffusion-xl-base-1.0"
        )
        torch_dtype = kwargs.pop("torch_dtype", torch.float16)
        unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path, torch_dtype=torch_dtype, subfolder="unet"
        ).to(device)

        if distri_config.parallelism == "patch":
            unet = DistriUNetPP(unet, distri_config)
        elif distri_config.parallelism == "tensor":
            unet = DistriUNetTP(unet, distri_config)
        elif distri_config.parallelism == "naive_patch":
            unet = NaivePatchUNet(unet, distri_config)
        else:
            raise ValueError(f"Unknown parallelism: {distri_config.parallelism}")

        if not kwargs.get("use_controlnet", False):
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                pretrained_model_name_or_path, torch_dtype=torch_dtype, unet=unet, **kwargs
            ).to(device)
        else:
            from diffusers import ControlNetModel
            controlnet = ControlNetModel.from_pretrained(
                "diffusers/controlnet-canny-sdxl-1.0",
                torch_dtype=torch.float16
            )
            controlnet.to(distri_config.device)

            if distri_config.parallelism == "patch":
                controlnet = DistriControlnetPP(controlnet, distri_config)

            pipeline = StableDiffusionXLPipelineControlNet.from_pretrained(
                pretrained_model_name_or_path, torch_dtype=torch_dtype, unet=unet, **kwargs
            ).to(device)
            pipeline.enable_controlnet( controlnet )
        return DistriSDXLPipeline(pipeline, distri_config)

    def set_progress_bar_config(self, **kwargs):
        self.pipeline.set_progress_bar_config(**kwargs)

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        assert "height" not in kwargs, "height should not be in kwargs"
        assert "width" not in kwargs, "width should not be in kwargs"
        config = self.distri_config
        if not config.do_classifier_free_guidance:
            if "guidance_scale" not in kwargs:
                kwargs["guidance_scale"] = 1
            else:
                assert kwargs["guidance_scale"] == 1
        self.pipeline.unet.set_counter(0)
        return self.pipeline(height=config.height, width=config.width, *args, **kwargs)

    @torch.no_grad()
    def prepare(self, **kwargs):
        distri_config = self.distri_config

        static_inputs = {}
        static_outputs = []
        cuda_graphs = []
        pipeline = self.pipeline

        height = distri_config.height
        width = distri_config.width
        assert height % 8 == 0 and width % 8 == 0

        original_size = (height, width)
        target_size = (height, width)
        crops_coords_top_left = (0, 0)

        device = distri_config.device

        prompt_embeds, _, pooled_prompt_embeds, _ = pipeline.encode_prompt(
            prompt="",
            prompt_2=None,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
            negative_prompt=None,
            negative_prompt_2=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
        )
        batch_size = 2 if distri_config.do_classifier_free_guidance else 1

        num_channels_latents = pipeline.unet.config.in_channels
        latents = pipeline.prepare_latents(
            batch_size, num_channels_latents, height, width, prompt_embeds.dtype, device, None
        )

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        if pipeline.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = pipeline.text_encoder_2.config.projection_dim

        add_time_ids = pipeline._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(1, 1)

        if batch_size > 1:
            prompt_embeds = prompt_embeds.repeat(batch_size, 1, 1)
            add_text_embeds = add_text_embeds.repeat(batch_size, 1)
            add_time_ids = add_time_ids.repeat(batch_size, 1)

        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
        t = torch.zeros([batch_size], device=device, dtype=torch.long)

        static_inputs["sample"] = latents
        static_inputs["timestep"] = t
        static_inputs["encoder_hidden_states"] = prompt_embeds
        static_inputs["added_cond_kwargs"] = added_cond_kwargs

        # static_inputs["down_block_additional_residuals"] = down_block_res_samples
        # create a zero tensor as a placeholder on the same device and torch.float16 with the same shape: (2, 1280, 32, 32)
        static_inputs["mid_block_additional_residual"] = torch.zeros([2, 1280, 32, 32], device=device, dtype=torch.float16)
        
        dummy_down_block_additional_residuals = []
        dummy_down_block_additional_residuals.append(torch.zeros([2, 320, 128, 128], device=device, dtype=torch.float16))
        dummy_down_block_additional_residuals.append(torch.zeros([2, 320, 128, 128], device=device, dtype=torch.float16))
        dummy_down_block_additional_residuals.append(torch.zeros([2, 320, 128, 128], device=device, dtype=torch.float16))
        dummy_down_block_additional_residuals.append(torch.zeros([2, 320, 64, 64], device=device, dtype=torch.float16))
        dummy_down_block_additional_residuals.append(torch.zeros([2, 640, 64, 64], device=device, dtype=torch.float16))
        dummy_down_block_additional_residuals.append(torch.zeros([2, 640, 64, 64], device=device, dtype=torch.float16))
        dummy_down_block_additional_residuals.append(torch.zeros([2, 640, 32, 32], device=device, dtype=torch.float16))
        dummy_down_block_additional_residuals.append(torch.zeros([2, 1280, 32, 32], device=device, dtype=torch.float16))
        dummy_down_block_additional_residuals.append(torch.zeros([2, 1280, 32, 32], device=device, dtype=torch.float16))
        assert len(dummy_down_block_additional_residuals) == 9, "dummy_down_block_additional_residuals should have 9 elements, but got {}".format(len(dummy_down_block_additional_residuals))
        static_inputs["down_block_additional_residuals"] = dummy_down_block_additional_residuals

        static_inputs_controlnet = {}
        static_inputs_controlnet["sample"] = deepcopy(latents)
        static_inputs_controlnet["timestep"] = deepcopy(t)
        static_inputs_controlnet["encoder_hidden_states"] = deepcopy(prompt_embeds)
        static_inputs_controlnet["added_cond_kwargs"] = deepcopy(added_cond_kwargs)
        controlnet_cond = torch.empty( (2, 3, 1024, 1024), dtype=torch.float16, device=device )

        # Used to create communication buffer
        comm_manager = None
        print("n_device_per_batch: ", distri_config.n_device_per_batch)
        if distri_config.n_device_per_batch > 1:
            comm_manager = PatchParallelismCommManager(distri_config)
            pipeline.unet.set_comm_manager(comm_manager)

            # Only used for creating the communication buffer
            pipeline.unet.set_counter(0)
            pipeline.unet(**static_inputs, return_dict=False, record=True)
            if comm_manager.numel > 0:
                comm_manager.create_buffer()
        
        if distri_config.n_device_per_batch > 1:
            comm_manager_controlnet = PatchParallelismCommManager(distri_config)
            pipeline.controlnet.set_comm_manager(comm_manager_controlnet)
            # Only used for creating the communication buffer
            pipeline.controlnet.set_counter(0)
            # pipeline.controlnet(**static_inputs, return_dict=False, record=True)
            pipeline.controlnet(
                static_inputs_controlnet["sample"],
                static_inputs_controlnet["timestep"],
                static_inputs_controlnet["encoder_hidden_states"],
                controlnet_cond,
                0.5,
                False,
                static_inputs_controlnet["added_cond_kwargs"],
                return_dict=False
            )
            if comm_manager_controlnet.numel > 0:
                comm_manager_controlnet.create_buffer()

        # Pre-run
        print("Pre-run")
        pipeline.unet.set_counter(0)
        pipeline.unet(**static_inputs, return_dict=False, record=True)

        if distri_config.use_cuda_graph:
            if comm_manager is not None:
                comm_manager.clear()
            if distri_config.parallelism == "naive_patch":
                counters = [0, 1]
            elif distri_config.parallelism == "patch":
                counters = [0, distri_config.warmup_steps + 1, distri_config.warmup_steps + 2]
            elif distri_config.parallelism == "tensor":
                counters = [0]
            else:
                raise ValueError(f"Unknown parallelism: {distri_config.parallelism}")
            print("counters: ", counters)
            for counter in counters:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    pipeline.unet.set_counter(counter)
                    output = pipeline.unet(**static_inputs, return_dict=False, record=True)[0]
                    static_outputs.append(output)
                cuda_graphs.append(graph)
            pipeline.unet.setup_cuda_graph(static_outputs, cuda_graphs)

        self.static_inputs = static_inputs


class DistriSDPipeline:
    def __init__(self, pipeline: StableDiffusionPipeline, module_config: DistriConfig):
        self.pipeline = pipeline
        self.distri_config = module_config

        self.static_inputs = None

        self.prepare()

    @staticmethod
    def from_pretrained(distri_config: DistriConfig, **kwargs):
        device = distri_config.device
        pretrained_model_name_or_path = kwargs.pop("pretrained_model_name_or_path", "CompVis/stable-diffusion-v1-4")
        torch_dtype = kwargs.pop("torch_dtype", torch.float16)
        unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path, torch_dtype=torch_dtype, subfolder="unet"
        ).to(device)

        if distri_config.parallelism == "patch":
            unet = DistriUNetPP(unet, distri_config)
        elif distri_config.parallelism == "tensor":
            unet = DistriUNetTP(unet, distri_config)
        elif distri_config.parallelism == "naive_patch":
            unet = NaivePatchUNet(unet, distri_config)
        else:
            raise ValueError(f"Unknown parallelism: {distri_config.parallelism}")

        pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path, torch_dtype=torch_dtype, unet=unet, **kwargs
        ).to(device)
        return DistriSDPipeline(pipeline, distri_config)

    def set_progress_bar_config(self, **kwargs):
        self.pipeline.set_progress_bar_config(**kwargs)

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        assert "height" not in kwargs, "height should not be in kwargs"
        assert "width" not in kwargs, "width should not be in kwargs"
        config = self.distri_config
        if not config.do_classifier_free_guidance:
            if not "guidance_scale" not in kwargs:
                kwargs["guidance_scale"] = 1
            else:
                assert kwargs["guidance_scale"] == 1
        self.pipeline.unet.set_counter(0)
        return self.pipeline(height=config.height, width=config.width, *args, **kwargs)

    @torch.no_grad()
    def prepare(self, **kwargs):
        distri_config = self.distri_config

        static_inputs = {}
        static_outputs = []
        cuda_graphs = []
        pipeline = self.pipeline

        height = distri_config.height
        width = distri_config.width
        assert height % 8 == 0 and width % 8 == 0

        device = distri_config.device

        prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(
            "",
            device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
            negative_prompt=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            lora_scale=None,
            clip_skip=kwargs.get("clip_skip", None),
        )

        batch_size = 2 if distri_config.do_classifier_free_guidance else 1

        num_channels_latents = pipeline.unet.config.in_channels
        latents = pipeline.prepare_latents(
            batch_size, num_channels_latents, height, width, prompt_embeds.dtype, device, None
        )

        prompt_embeds = prompt_embeds.to(device)

        if batch_size > 1:
            prompt_embeds = prompt_embeds.repeat(batch_size, 1, 1)

        t = torch.zeros([batch_size], device=device, dtype=torch.long)

        static_inputs["sample"] = latents
        static_inputs["timestep"] = t
        static_inputs["encoder_hidden_states"] = prompt_embeds

        # Used to create communication buffer
        comm_manager = None
        if distri_config.n_device_per_batch > 1:
            comm_manager = PatchParallelismCommManager(distri_config)
            pipeline.unet.set_comm_manager(comm_manager)

            # Only used for creating the communication buffer
            pipeline.unet.set_counter(0)
            pipeline.unet(**static_inputs, return_dict=False, record=True)
            if comm_manager.numel > 0:
                comm_manager.create_buffer()

        # Pre-run
        pipeline.unet.set_counter(0)
        pipeline.unet(**static_inputs, return_dict=False, record=True)

        if distri_config.use_cuda_graph:
            if comm_manager is not None:
                comm_manager.clear()
            if distri_config.parallelism == "naive_patch":
                counters = [0, 1]
            elif distri_config.parallelism == "patch":
                counters = [0, distri_config.warmup_steps + 1, distri_config.warmup_steps + 2]
            elif distri_config.parallelism == "tensor":
                counters = [0]
            else:
                raise ValueError(f"Unknown parallelism: {distri_config.parallelism}")
            for counter in counters:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    pipeline.unet.set_counter(counter)
                    output = pipeline.unet(**static_inputs, return_dict=False, record=True)[0]
                    static_outputs.append(output)
                cuda_graphs.append(graph)
            pipeline.unet.setup_cuda_graph(static_outputs, cuda_graphs)

        self.static_inputs = static_inputs
