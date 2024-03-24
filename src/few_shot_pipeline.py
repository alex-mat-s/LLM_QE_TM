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
            data: pd.core.frame.DataFrame,
            model_name: str = 'Mistral',
            quantization_mode: str = '4_bit',  # Only for Mistral
            device: str = 'cpu',
            system_prompt: str = '',
            task: str = '',
            file:str = '', # The full path
            print_final_prompt: bool = False,
            num_iter: int = None  # The number of samples (if the user doesn't want to process the entire dataset)
        ):
        
        # Assertions
        # Check data type
        assertIsInstance(data, pd.core.frame.DataFrame, "Incorrect data type, it must be pandas.core.frame.DataFrame")
        assertFalse(task, "The task is not specified")
        assert model_name in ["mistral", "llama"]
        assertNotEqual(data.columns, ["wordset", "true"])
        # Проверка на соответствие именованиям колонок!


        self.device = device
        self.data = data
        
        if model_name == "mistral":
            self.model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        # Load Model
        self.model = self.__set_model(quantization_mode)
        # Load tokenizer
        self.tokenizer = self.__set_tokenizer()
        self.task = task
        
        if system_prompt:
            self.system_prompt = system_prompt
        else:
            self.system_prompt = self.get_system_prompt()

        self.res_file = file
        self.print_final_prompt = print_final_prompt
        
        self.size = self.get_dataset_size() if num_iter is None else num_iter

    def __set_model(self, quantization_mode):
        if quantization_mode == "4_bit":
            # Qunatization config
            quantization_config_4bit = BitsAndBytesConfig(
                load_in_4bit = True,  # enable 4-bit quantization
                bnb_4bit_quant_type = 'nf4',  # information theoretically optimal dtype for normally distributed weights
                bnb_4bit_use_double_quant = True,  # quantize quantized weights //insert xzibit meme
                bnb_4bit_compute_dtype = torch.bfloat16  # optimized fp format for ML
            )  

            return AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config_4bit
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
            

        
    
        
        


