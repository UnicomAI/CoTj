
import numpy as np
from PIL import Image
from utils import *
import torch
from diffusers import ZImagePipeline
import os

class CoTjZImageTurboPipeline():
    def __init__(self, model_path='~/Tongyi-MAI/Z-Image-Turbo/',
                  mlp_path='./', pipe=None, width=1024, height=1024, device='cuda:0') -> None:
        """
        Initialize the text-image model.
        
        Args:
            model_path (str): The path to the model (default is "Qwen/Qwen-Image")
            device (Optional[str]): The device to use ('cuda' or 'cpu'). Auto-detects if None.
        """

        #pipe load...
        print('pipe loading')
        self.torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.device = device if device is not None and torch.cuda.is_available() else "cpu"
        if pipe is None:
            pipe = ZImagePipeline.from_pretrained(model_path, 
                                                torch_dtype=self.torch_dtype,
                                                low_cpu_mem_usage=False)
        self.pipe = pipe.to(self.device)

        
        print('mlp model loading')
        self.mlp_model = SimpleMLP(2560,100)
        self.mlp_model.load_state_dict(torch.load(os.path.join(mlp_path,'model_state.pth')))
        self.mlp_model.to(self.device)
        self.mlp_model.eval()

        norm_data = torch.load(os.path.join(mlp_path,'norm_data.pt'))
        self.mean_dna  = norm_data['Y_mean'].view(1,-1).float().to(self.device)
        self.mean_embeds  = norm_data['X_mean'].view(1,-1).float().to(self.device)
        self.std_embeds  = norm_data['X_std'].view(1,-1).float().to(self.device)
        
    


        self.width = width
        self.height = height
        self.mu = None
        self.mu = self.calc_mu(width = self.width, height=self.height) #1.15
        
        self.scheduler_config = {
            "num_train_timesteps": self.pipe.scheduler.config.num_train_timesteps,
            "shift": self.pipe.scheduler.config.shift,
            "use_dynamic_shifting": self.pipe.scheduler.config.use_dynamic_shifting,
            "base_shift": self.pipe.scheduler.config.base_shift,
            "max_shift": self.pipe.scheduler.config.max_shift,
            "base_image_seq_len": self.pipe.scheduler.config.base_image_seq_len,
            "max_image_seq_len": self.pipe.scheduler.config.max_image_seq_len,
            "invert_sigmas": self.pipe.scheduler.config.invert_sigmas,
            "shift_terminal": self.pipe.scheduler.config.shift_terminal,
            "use_karras_sigmas": self.pipe.scheduler.config.use_karras_sigmas,
            "use_exponential_sigmas": self.pipe.scheduler.config.use_exponential_sigmas,
            "use_beta_sigmas": self.pipe.scheduler.config.use_beta_sigmas,
            "time_shift_type": self.pipe.scheduler.config.time_shift_type,
            "stochastic_sampling": self.pipe.scheduler.config.stochastic_sampling
        }

        print('mu:', self.mu, 'scheduler_config is ready.')
    
    def align_dna(self, dna, ref_dna=None):
       
        if ref_dna is None:
            ref_dna = self.mean_dna.view(-1).cpu().numpy()

        a = np.array(ref_dna)
        b = np.array(dna)
        
        std_a = np.std(a)
       
        std_b = np.std(b)
       
        if std_b < 1e-12:
            return np.full_like(b, np.mean(a))

        scale = std_a / std_b
        aligned_b = b * scale

        aligned_b = aligned_b - aligned_b.min() + a.min()

        return aligned_b
    
    def calculate_shift(
        self,
        image_seq_len: int = 4096,
        base_seq_len: int = 256,
        max_seq_len: int = 4096,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
        ):
        base_seq_len = self.pipe.scheduler.config.get("base_image_seq_len", 256)
        max_seq_len  = self.pipe.scheduler.config.get("max_image_seq_len", 4096)
        base_shift   = self.pipe.scheduler.config.get("base_shift", 0.5)
        max_shift    = self.pipe.scheduler.config.get("max_shift", 1.15)
                
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return mu 


    def get_prompts_embeds(self, prompt='', height=None, width=None):
        prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
                    prompt=prompt,
                    negative_prompt=None,
                    do_classifier_free_guidance=self.pipe.do_classifier_free_guidance,
                    prompt_embeds=None,
                    negative_prompt_embeds=None,
                    device=self.device,
                    max_sequence_length=512,
                )
        return prompt_embeds
    

    def calc_mu(self, width=1024, height=1024):
        if width!=self.width or height!=self.height or self.mu is None:
            print('calc mu....')
            num_channels_latents = self.pipe.transformer.in_channels
            noise_latent = self.pipe.prepare_latents(
                    1,
                    num_channels_latents,
                    height,
                    width,
                    self.torch_dtype,
                    self.device,
                    generator=torch.Generator(device=self.device).manual_seed(-1),
                )
            image_seq_len = (noise_latent.shape[2] // 2) * (noise_latent.shape[3] // 2)
            mu = self.calculate_shift(image_seq_len = image_seq_len)
        else:
            mu = self.mu
        return mu

    @torch.no_grad()    
    def gen_optimal_image(self, prompt=None, times_optimal=None, width=1024, height=1024,seed=42, prompt_embeds=None):
        mu = self.calc_mu(width=height,height=height)
        self.pipe.scheduler.set_timesteps(timesteps=torch.tensor(times_optimal)*1000, mu=mu)
        inverse_sigmas = inverse_euler_set_timesteps(self.pipe.scheduler.sigmas.clone(), 
                                                        scheduler_config=self.scheduler_config, 
                                                        mu=mu)
        image = self.pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_images_per_prompt=1,
            sigmas = inverse_sigmas,
            guidance_scale=0.0,
            prompt_embeds = prompt_embeds,
            generator = torch.Generator(device=self.device).manual_seed(seed),
            output_type = 'pil'
        )

        return image
              
    def get_pipe_image(self, prompt, num_inference_steps=8, width=1024, height=1024,seed=42):
        image = self.pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_images_per_prompt=1,
                num_inference_steps =num_inference_steps, #4 and 8
                guidance_scale=0.0,     # Guidance should be 0 for the Turbo models
                generator = torch.Generator(device=self.device).manual_seed(seed),
                output_type = 'pil'
            )
        return image


    @torch.no_grad()
    def get_dna(self, prompt, width=1664, height=928):
        embed = self.get_prompts_embeds(prompt = prompt, width=width, height=height)
        new_input_raw = embed[0].mean(0).view(1,-1).float().to(self.device)
        new_input_raw = (new_input_raw - self.mean_embeds) / (self.std_embeds + 1e-9)
        dna_predicted = self.mlp_model(new_input_raw)
        prompt_dna = dna_predicted.cpu()

        dna_list = [k.item() for k in prompt_dna.view(-1)]
        dna_list = self.align_dna(dna_list)
        return dna_list, embed
    
    @torch.no_grad()
    def get_prompt_cotj_image_fixed_step(self, 
                                    prompt, 
                                    num_inference_steps=8, 
                                    width=1024, 
                                    height=1024,
                                    seed=42):
        dna_list, embed = self.get_dna(prompt, width, height)
        prompt_graph = GraphSearch(dna_list,super_k=None)
        cost, path, times_optimal = prompt_graph.find_optimal_k_times(num_inference_steps)
        del prompt_graph
        return self.gen_optimal_image(prompt = None, 
                                    times_optimal = times_optimal, 
                                    width=width, 
                                    height=height, 
                                    seed=seed, 
                                    prompt_embeds = embed)
    
    @torch.no_grad()
    def get_prompt_cotj_image_adaptive_step(self, 
                                    prompt, 
                                    inference_steps_max = 8, 
                                    fidelity_target=0.99, 
                                    width=1024, 
                                    height=1024,
                                    seed=42):
        dna_list, embed = self.get_dna(prompt, width, height)
        prompt_graph = GraphSearch(dna_list,super_k=None)
        cost, path, times_optimal = prompt_graph.find_optimal_adaptive_times(inference_steps_max, fidelity_target)
        del prompt_graph
        return self.gen_optimal_image(prompt = None, 
                                    times_optimal = times_optimal, 
                                    width=width, 
                                    height=height, 
                                    seed=seed, 
                                    prompt_embeds = embed)
      






if __name__ == "__main__":
    model_path = '~/Tongyi-MAI/Z-Image-Turbo/'
    mlp_path = './prompt_models/zimageturbo_mlp_models/'
    device='cuda:1'
    pipe = None
    cotj = CoTjQwenImagePipeline(model_path=model_path,mlp_path=mlp_path, pipe=pipe, device=device)

    prompt ="a peony flower."

    num_inference_steps = 10

    #baseline euler.
    pipe_image  = cotj.get_pipe_image(prompt, 
                                      num_inference_steps=num_inference_steps, 
                                      width=1024, 
                                      height=1024,
                                      seed=42)
    
    #Fixed-Step Planning
    prompt_cotj_image_fixed = cotj.get_prompt_cotj_image_fixed_step(prompt, 
                                                        num_inference_steps=num_inference_steps, 
                                                        width=1024, 
                                                        height=1024,
                                                        seed=42)
    #Adaptive-Length Planning
    prompt_cotj_image_adaptive = cotj.get_prompt_cotj_image_adaptive_step(prompt, 
                                        inference_steps_max = 8, 
                                        fidelity_target=0.99, 
                                        width=1024, 
                                        height=1024,
                                        seed=42)