import torch
from diffusers import UNet2DConditionModel
from diffusers.models.attention_processor import Attention
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from torch import distributed as dist, nn

from .base_model import BaseModel
from distrifuser.modules.pp.attn import DistriCrossAttentionPP, DistriSelfAttentionPP
from distrifuser.modules.base_module import BaseModule
from distrifuser.modules.pp.conv2d import DistriConv2dPP
from distrifuser.modules.pp.groupnorm import DistriGroupNorm
from ..utils import DistriConfig


class DistriUNetPP(BaseModel):  # for Patch Parallelism
    def __init__(self, model: UNet2DConditionModel, distri_config: DistriConfig):
        assert isinstance(model, UNet2DConditionModel)
        if distri_config.world_size > 1 and distri_config.n_device_per_batch > 1:
            for name, module in model.named_modules():
                if isinstance(module, BaseModule):
                    continue
                for subname, submodule in module.named_children():
                    if isinstance(submodule, nn.Conv2d):
                        kernel_size = submodule.kernel_size
                        if kernel_size == (1, 1) or kernel_size == 1:
                            continue
                        wrapped_submodule = DistriConv2dPP(
                            submodule, distri_config, is_first_layer=subname == "conv_in"
                        )
                        setattr(module, subname, wrapped_submodule)
                    elif isinstance(submodule, Attention):
                        if subname == "attn1":  # self attention
                            wrapped_submodule = DistriSelfAttentionPP(submodule, distri_config)
                        else:  # cross attention
                            assert subname == "attn2"
                            wrapped_submodule = DistriCrossAttentionPP(submodule, distri_config)
                        setattr(module, subname, wrapped_submodule)
                    elif isinstance(submodule, nn.GroupNorm):
                        wrapped_submodule = DistriGroupNorm(submodule, distri_config)
                        setattr(module, subname, wrapped_submodule)

        super(DistriUNetPP, self).__init__(model, distri_config)

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: torch.Tensor or float or int,
        encoder_hidden_states: torch.Tensor,
        class_labels: torch.Tensor or None = None,
        timestep_cond: torch.Tensor or None = None,
        attention_mask: torch.Tensor or None = None,
        cross_attention_kwargs: dict[str, any] or None = None,
        added_cond_kwargs: dict[str, torch.Tensor] or None = None,
        down_block_additional_residuals: tuple[torch.Tensor] or None = None,
        mid_block_additional_residual: torch.Tensor or None = None,
        down_intrablock_additional_residuals: tuple[torch.Tensor] or None = None,
        encoder_attention_mask: torch.Tensor or None = None,
        return_dict: bool = True,
        record: bool = False,
    ):
        distri_config = self.distri_config
        b, c, h, w = sample.shape
        assert (
            class_labels is None
            and timestep_cond is None
            and attention_mask is None
            and cross_attention_kwargs is None
            and down_intrablock_additional_residuals is None
            and encoder_attention_mask is None
        )

        # print("distri_config.use_cuda_graph", distri_config.use_cuda_graph, "record", record)
        if distri_config.use_cuda_graph and not record:
            static_inputs = self.static_inputs

            if distri_config.world_size > 1 and distri_config.do_classifier_free_guidance and distri_config.split_batch:
                assert b == 2
                batch_idx = distri_config.batch_idx()
                sample = sample[batch_idx : batch_idx + 1]
                timestep = (
                    timestep[batch_idx : batch_idx + 1] if torch.is_tensor(timestep) and timestep.ndim > 0 else timestep
                )
                encoder_hidden_states = encoder_hidden_states[batch_idx : batch_idx + 1]
                if added_cond_kwargs is not None:
                    for k in added_cond_kwargs:
                        added_cond_kwargs[k] = added_cond_kwargs[k][batch_idx : batch_idx + 1]

            assert static_inputs["sample"].shape == sample.shape
            static_inputs["sample"].copy_(sample)
            if torch.is_tensor(timestep):
                if timestep.ndim == 0:
                    for b in range(static_inputs["timestep"].shape[0]):
                        static_inputs["timestep"][b] = timestep.item()
                else:
                    assert static_inputs["timestep"].shape == timestep.shape
                    static_inputs["timestep"].copy_(timestep)
            else:
                for b in range(static_inputs["timestep"].shape[0]):
                    static_inputs["timestep"][b] = timestep
            assert static_inputs["encoder_hidden_states"].shape == encoder_hidden_states.shape
            static_inputs["encoder_hidden_states"].copy_(encoder_hidden_states)
            if added_cond_kwargs is not None:
                for k in added_cond_kwargs:
                    assert static_inputs["added_cond_kwargs"][k].shape == added_cond_kwargs[k].shape
                    static_inputs["added_cond_kwargs"][k].copy_(added_cond_kwargs[k])

            if mid_block_additional_residual is not None:
                assert static_inputs["mid_block_additional_residual"].shape == mid_block_additional_residual.shape, f"{static_inputs['mid_block_additional_residual'].shape} != {mid_block_additional_residual.shape}"
                static_inputs["mid_block_additional_residual"].copy_(mid_block_additional_residual)
                
                for i in range(len(down_block_additional_residuals)):
                    assert static_inputs["down_block_additional_residuals"][i].shape == down_block_additional_residuals[i].shape, f"{static_inputs['down_block_additional_residuals'][i].shape} != {down_block_additional_residuals[i].shape}"
                    static_inputs["down_block_additional_residuals"][i].copy_(down_block_additional_residuals[i])

            if self.counter <= distri_config.warmup_steps:
                graph_idx = 0
            elif self.counter == distri_config.warmup_steps + 1:
                graph_idx = 1
            else:
                graph_idx = 2

            self.cuda_graphs[graph_idx].replay()
            output = self.static_outputs[graph_idx]
        else:
            if distri_config.world_size == 1:
                output = self.model(
                    sample,
                    timestep,
                    encoder_hidden_states,
                    class_labels=class_labels,
                    timestep_cond=timestep_cond,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    down_block_additional_residuals=down_block_additional_residuals,
                    mid_block_additional_residual=mid_block_additional_residual,
                    down_intrablock_additional_residuals=down_intrablock_additional_residuals,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
            elif distri_config.do_classifier_free_guidance and distri_config.split_batch:
                assert b == 2
                batch_idx = distri_config.batch_idx()
                # print("batch_idx", batch_idx, "distri_config.rank", distri_config.rank)
                sample = sample[batch_idx : batch_idx + 1]
                timestep = (
                    timestep[batch_idx : batch_idx + 1] if torch.is_tensor(timestep) and timestep.ndim > 0 else timestep
                )
                encoder_hidden_states = encoder_hidden_states[batch_idx : batch_idx + 1]
                if added_cond_kwargs is not None:
                    new_added_cond_kwargs = {}
                    for k in added_cond_kwargs:
                        new_added_cond_kwargs[k] = added_cond_kwargs[k][batch_idx : batch_idx + 1]
                    added_cond_kwargs = new_added_cond_kwargs
                if distri_config.world_size == 2:
                    mid_block_additional_residual = mid_block_additional_residual[batch_idx : batch_idx + 1]
                    down_block_additional_residuals = [
                        residual[batch_idx : batch_idx + 1] for residual in down_block_additional_residuals
                    ]

                if distri_config.world_size < 4:
                    output = self.model(
                        sample,
                        timestep,
                        encoder_hidden_states,
                        class_labels=class_labels,
                        timestep_cond=timestep_cond,
                        attention_mask=attention_mask,
                        cross_attention_kwargs=cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        down_block_additional_residuals=down_block_additional_residuals,
                        mid_block_additional_residual=mid_block_additional_residual,
                        down_intrablock_additional_residuals=down_intrablock_additional_residuals,
                        encoder_attention_mask=encoder_attention_mask,
                        return_dict=False,
                    )[0]
                elif distri_config.world_size == 4 or distri_config.world_size == 8:
                    mid_block_res_dim2 = mid_block_additional_residual.shape[2] // distri_config.n_device_per_batch
                    mid_block_additional_residual_split = mid_block_additional_residual[batch_idx : batch_idx + 1, :, batch_idx*mid_block_res_dim2:(batch_idx+1)*mid_block_res_dim2, :]
                    
                    down_block_additional_residuals_split = []
                    for i in range(len(down_block_additional_residuals)):
                        res_dim2 = down_block_additional_residuals[i].shape[2] // distri_config.n_device_per_batch
                        down_block_additional_residuals_split.append(down_block_additional_residuals[i][batch_idx : batch_idx + 1, :, batch_idx*res_dim2:(batch_idx+1)*res_dim2, :])

                    # for i in range(len(down_block_additional_residuals)):
                    #     res_dim2 = down_block_additional_residuals[i].shape[2] // 2
                    #     down_block_additional_residuals[i] = down_block_additional_residuals[i][batch_idx : batch_idx + 1, :, batch_idx*res_dim2:(batch_idx+1)*res_dim2, :]
                    
                    output = self.model(
                        sample,
                        timestep,
                        encoder_hidden_states,
                        class_labels=class_labels,
                        timestep_cond=timestep_cond,
                        attention_mask=attention_mask,
                        cross_attention_kwargs=cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        down_block_additional_residuals=down_block_additional_residuals_split,
                        mid_block_additional_residual=mid_block_additional_residual_split,
                        down_intrablock_additional_residuals=down_intrablock_additional_residuals,
                        encoder_attention_mask=encoder_attention_mask,
                        return_dict=False,
                    )[0]

                if self.output_buffer is None:
                    self.output_buffer = torch.empty((b, c, h, w), device=output.device, dtype=output.dtype)
                if self.buffer_list is None:
                    self.buffer_list = [torch.empty_like(output) for _ in range(distri_config.world_size)]
                dist.all_gather(self.buffer_list, output.contiguous(), async_op=False)
                torch.cat(self.buffer_list[: distri_config.n_device_per_batch], dim=2, out=self.output_buffer[0:1])
                torch.cat(self.buffer_list[distri_config.n_device_per_batch :], dim=2, out=self.output_buffer[1:2])
                output = self.output_buffer
            else:
                output = self.model(
                    sample,
                    timestep,
                    encoder_hidden_states,
                    class_labels=class_labels,
                    timestep_cond=timestep_cond,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    down_block_additional_residuals=down_block_additional_residuals,
                    mid_block_additional_residual=mid_block_additional_residual,
                    down_intrablock_additional_residuals=down_intrablock_additional_residuals,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
                if self.output_buffer is None:
                    self.output_buffer = torch.empty((b, c, h, w), device=output.device, dtype=output.dtype)
                if self.buffer_list is None:
                    self.buffer_list = [torch.empty_like(output) for _ in range(distri_config.world_size)]
                output = output.contiguous()
                dist.all_gather(self.buffer_list, output, async_op=False)
                torch.cat(self.buffer_list, dim=2, out=self.output_buffer)
                output = self.output_buffer
            if record:
                if self.static_inputs is None:
                    if distri_config.world_size == 4 or distri_config.world_size == 8:
                        self.static_inputs = {
                            "sample": sample,
                            "timestep": timestep,
                            "encoder_hidden_states": encoder_hidden_states,
                            "added_cond_kwargs": added_cond_kwargs,
                            "mid_block_additional_residual": mid_block_additional_residual_split,
                            "down_block_additional_residuals": down_block_additional_residuals_split,
                        }
                        # print("sample shape", sample.shape)
                        # print("mid_block_additional_residual shape", mid_block_additional_residual_split.shape)
                        # print("down_block_additional_residuals[0] shape", down_block_additional_residuals_split[0].shape)
                    else:
                        self.static_inputs = {
                            "sample": sample,
                            "timestep": timestep,
                            "encoder_hidden_states": encoder_hidden_states,
                            "added_cond_kwargs": added_cond_kwargs,
                            "mid_block_additional_residual": mid_block_additional_residual,
                            "down_block_additional_residuals": down_block_additional_residuals,
                        }
                self.synchronize()

        if return_dict:
            output = UNet2DConditionOutput(sample=output)
        else:
            output = (output,)

        self.counter += 1
        return output

    @property
    def add_embedding(self):
        return self.model.add_embedding
