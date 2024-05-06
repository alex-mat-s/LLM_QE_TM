#!/usr/bin/env python3
import unittest
import torch
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


class TmEval:
    def __init__(
            self,
            model: str,
            quantization_mode: bool = False,  # Only for Mistral
            device: str = 'cpu'
        ):
        
        self.device = device
        
        # Load Model
        self.model = self.__set_model(quantization_mode)
        # Load tokenizer
        self.tokenizer = self.__set_tokenizer()
        
    def __set_model(self, quantization_mode):
        if quantization_mode:
            # Qunatization config
            quantization_config_4bit = BitsAndBytesConfig(
                load_in_4bit = True,  # enable 4-bit quantization
                bnb_4bit_quant_type = 'nf4',  # information theoretically optimal dtype for normally distributed weights
                bnb_4bit_use_double_quant = True,  # quantize quantized weights
                bnb_4bit_compute_dtype = torch.bfloat16  # optimized fp format for ML
            )  

            return AutoModelForCausalLM.from_pretrained(
                self.model,
                quantization_config=quantization_config_4bit
            )
        else:
            return AutoModelForCausalLM.from_pretrained(
                self.model,
                device_map="auto",
                trust_remote_code=False,
                revision="main"
            )

    def __set_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_name)
    
    def get_dataset_size(self):
        return self.data.shape[0]
    
    def get_system_prompt(self):
        pass

    def get_prompt(self):
        if self.task == "classification":
            if "mistral" in self.model_name:
                pass

    def get_results(self):
        # Remove old file
        if os.path.exists(self.res_file):
            os.remove(self.res_file)

        for i in tqdm(range(self.size)):
            sample = self.data.iloc[i]["wordset"]
            if self.task == "classification":
                prompt = self.get_prompt()        
            

        
    
        
        


