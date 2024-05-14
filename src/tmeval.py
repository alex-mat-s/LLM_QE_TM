import torch
import os
import re
import warnings
import phik
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

from pathlib import Path
from tqdm import tqdm
from typing import Union
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

import data_preprocessing 
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

class TmEval:
    def __init__(
            self,
            model_name: str,
            quantization_mode: Union[str, bool] = False,  # Only for Mistral
            device: str = 'cpu'
        ):

        if isinstance(quantization_mode, str):
            assert device == 'cuda', 'If you want to use quantization mode for your LLM, please, provide GPU'
        self.device = device

        # Load Model
        self.model = self.__set_model(quantization_mode, model_name)
        # Load tokenizer
        self.tokenizer = self.__set_tokenizer(model_name)

    def __set_model(self, quantization_mode, model_name):
        if quantization_mode == 'nf4':
            # Qunatization config
            quantization_config_4bit = BitsAndBytesConfig(
                load_in_4bit = True,  # enable 4-bit quantization
                bnb_4bit_quant_type = 'nf4',  # information theoretically optimal dtype for normally distributed weights
                bnb_4bit_use_double_quant = True,  # quantize quantized weights
                bnb_4bit_compute_dtype = torch.bfloat16  # optimized fp format for ML
            )

            return AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config_4bit
            )
        else:
            return AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=self.device,
                trust_remote_code=False,
                revision="main"
            )

    def __set_tokenizer(self, model_name):
        return AutoTokenizer.from_pretrained(model_name)

    def get_dataset_size(self):
        return self.data.shape[0]

    def generate_answer(self, prompt, temp, n_token):
        torch.cuda.empty_cache()
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        model_input = encoded
        model_input = model_input.to(self.device)
        if self.device=='cpu':
            self.model.to(self.device)
        generated_ids = self.model.generate(
            **model_input, do_sample=True,
            max_new_tokens=n_token,
            temperature=temp,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id
            )
        decoded = self.tokenizer.batch_decode(generated_ids)
        return decoded

    def extract_substring(self, input_string):
        index = input_string.find("[/INST]")
        if index != -1:
            return input_string[index + len("[/INST]"):]
        else:
            return ""

    def get_answer(
            self,
            temp=0.1,
            n_token=500,
            instruction='',
            sample='',
            use_system_tokens=True,
            print_prompt=False,
            system_prompt=""
        ):
        """
        Get result of the inference for one sample.

        Parameters
        ----------
        temp : float, default=0.1
            The value used to modulate the next token probabilities.
        n_token : int, default=500
            The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
        instruction : str
            The instruction part of the prompt.
        sample : str
            The wordset
        use_system_tokens : bool, dafault=True
            If True, then special tokens will be used.
        print_prompt : bool, default=False
            Print the generated prompt.
        system_prompt : str
            Use system prompt (please, use this prompt for Llama model!).

        Returns
        -------
        result : str
            Generated response.
        """

        # Construct prompt
        if use_system_tokens:
            model_name = self.model.config._name_or_path.lower()
            pattern = r"(mistral|llama|gpt)"
            model_name = re.findall(pattern, model_name)[0]

            if model_name == "mistral":
                prompt = f"<s> [INST] {instruction} {sample} [/INST]"
            elif model_name == "llama":
                if len(system_prompt) == 0:
                    prompt = f"<s>[INST]{instruction}{sample}[/INST]"
                else:                    
                    prompt = f"<s> [INST] <<SYS>> {system_prompt} <</SYS>> {instruction} {sample} [/INST]"
            elif model_name == "gpt":
                print("There is no system token for GPT. The 'without using system tokens' mode is enabled. Internal algorithms for postprocessing and calculation of metrics can give unstable results")
                prompt = f"{instruction}\n{sample}"
        else:
            warnings.warn("""The "without using system tokens" mode is enabled. Internal algorithms for postprocessing and calculation of metrics can give unstable results""")
            prompt = f"{instruction}\n{sample}"

        if print_prompt:
            print(prompt)

        # Generate answer
        answer = self.generate_answer(prompt, temp, n_token)

        # Extract the result
        if use_system_tokens:
            result = self.extract_substring(answer[0])
        else:
            result = answer[0]
        return result.replace("\n", " ")

    def get_prompt(self, prompt_type, sample):
        try:
            prompt_type is not None
        except:
            print("Please, provide the type of the internal prompt or ")
        system_prompt = ''
        instruction = ''
        if prompt_type == "P1_Mistral":
            instruction = ("You are a useful assistant who evaluates the coherence of words.\n"
                    "You will receive a list of words, please determine which class the given "
                    "list of words belongs to by answering the question: 'Is it possible to determine "
                    "a common topic for the presented word set or at least for the most part of the set?'."
                    "Classification rules: yes - if words have a strong connection between them, "
                    "rather yes - if some words are too common or out of topic, "
                    "rather no - if the amount of irrelevant words is high to determine a topic or there is a mixture of topics, "
                    "no - when words seem to be unconnected, "
                    "neutral - if it is hard for you to answer on the question.\nPrint only class without explanation and additional information.\n")

            sample = 'Words: ' + sample + "\nClass:"
        elif prompt_type == "P1_Llama":
            system_prompt = "You are a useful assistant who evaluates the coherence of words."
            instruction = ("You will receive a list of words, please determine which class the given "
                    "list of words belongs to by answering the question: 'Is it possible to determine "
                    "a common topic for the presented word set or at least for the most part of the set?'."
                    "Classification rules: yes - if words have a strong connection between them, "
                    "rather yes - if some words are too common or out of topic, "
                    "rather no - if the amount of irrelevant words is high to determine a topic or there is a mixture of topics, "
                    "no - when words seem to be unconnected, "
                    "neutral - if it is hard for you to answer on the question.\nPrint only class without explanation and additional information.\n")

            sample = 'Words: ' + sample + "\nClass:"
        elif prompt_type == "C2_Llama":
            system_prompt = ""
            instruction = ("You are the assistant for text classification. You will receive a TEXT, and you should answer 'YES' or 'NO' to the question: "
                           "'Is it possible to determine a common topic for the TEXT or at least for the most part of the TEXT?'. "
                           "Please, make sure you to only return YES or NO and nothing more.\n")
            sample = "TEXT: " + sample + "\nANSWER:"
        elif prompt_type == "C4_Llama":
            system_prompt = ""
            instruction = ("You are the assistant for text classification. "
                           "You will receive a list of words, please determine which class the given "
                           "list of words belongs to by answering the question: 'Is it possible to determine "
                           "a common topic for the presented word set or at least for the most part of the set?'. "
                           "Classification rules:yes - if words have a strong connection between them; "
                           "rather yes - if some words are too common or out of topic; "
                           "rather no - if the amount of irrelevant words is high to determine a topic or there is a mixture of topics; "
                           "no - when words seem to be unconnected. "
                           "Print only class without explanation and additional information.\n")
            sample = "Words: " + sample + "\nClass:"
        elif prompt_type == "C5_Llama":
            system_prompt = ""
            instruction = ("You will receive a list of words, please determine which class the given "
                           "list of words belongs to by answering the question: 'Is it possible to determine "
                           "a common topic for the presented word set or at least for the most part of the set?'. "
                           "Classification rules:yes - if words have a strong connection between them; "
                           "rather yes - if some words are too common or out of topic; "
                           "rather no - if the amount of irrelevant words is high to determine a topic or there is a mixture of topics; "
                           "no - when words seem to be unconnected; "
                           "neutral - if it is hard for you to answer on the question. "
                           "Print only class without explanation and additional information.\n")
            sample = "Words: " + sample + "\nClass:"
        # elif prompt_type == "TG_Llama":
        #     system_prompt
            

        return system_prompt, instruction, sample

    def get_results(
            self,
            data,
            filename,
            num_iter,
            base_dir,
            temp=0.1,
            prompt_type=None,
            n_token=500,
            remove_old_file=True,
            print_info=False,
            use_system_tokens=True,
            save_results_to_df=False
        ):
        """
        Get inference on the dataset.

        Parameters
        ----------
        data : pd.DataFrame
            Input dataset.
        filename : str
            The file where the results of the generation will be recorded.
        num_iter : int
            The number of samples.
        temp : float, default=0.1
            The value used to modulate the next token probabilities.
        prompt_type : None, str, default=None
            The type of the standard prompt.
        n_token : int, default=500
            The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
        remove_old_file : bool, default=True
            If True, then an old file with results will be removed.
        print_info : bool, default=False
            Print the generated prompt and the result of the inference.
        use_system_tokens : bool, default=True
            If True, then special tokens will be used.
        save_results_to_df : bool, default=False
            Save the results of the inference to the dataframe.
        """

        # Check if the columns of the dataset have appropriate names
        try:
            data[["task", "agg_label"]]
        except:
            print("There are no 'task' and 'agg_label' columns in the dataset!")

        # Create file
        data_dir = base_dir.joinpath('data/results')
        res_file = data_dir.joinpath(filename)

        # Remove old file
        if remove_old_file:
            if os.path.exists(res_file):
                os.remove(res_file)

        model_lst = []
        true_lst = []
        wordset_lst = []
        for i in tqdm(range(num_iter)):
            sample = data.iloc[i]["task"]

            system_prompt, instruction, sample = self.get_prompt(prompt_type, sample)

            res_true = data.iloc[i]["agg_label"]
            res_model = self.get_answer(
                temp=temp,
                n_token=n_token,
                instruction=instruction,
                sample=sample,
                use_system_tokens=True,
                print_prompt=print_info,
                system_prompt=system_prompt
            )
            if print_info:
                print(res_model)

            with open(res_file, 'a') as file:
                file.write(f"{data.iloc[i]['task']}\t{res_true}\t{res_model}\n")

            if save_results_to_df:
                model_lst.append(res_model)
                true_lst.append(res_true)
                wordset_lst.append(data.iloc[i]["task"])

        print(f"\nAll results were saved to the {res_file} file.")
        if save_results_to_df:
            print("All results were saved to the dataframe")
            data_res = pd.DataFrame({
                "task": wordset_lst,
                "true": true_lst,
                "model": model_lst}
            )
            return data_res

    def num_to_words(x):
        class_labels = {
            -2: 'bad',
            -1: 'rather_bad',
            0: 'neutral',
            1: 'rather_good',
            2: 'good'
        }

        return class_labels[x]

    def words_to_num(x):
        class_labels = {
            'bad': -2,
            'rather_bad': -1,
            'neutral': 0,
            'rather_good': 1,
            'good': 2
        }
        return class_labels[x]

    @classmethod
    def extract_answer(cls, input_string, num_class, answer_type="word"):
        # Lowercase
        res_answer = input_string.lower()
        # Remove extra spaces
        res_answer = res_answer.strip()

        if answer_type == "word":
            if num_class == 2:
                res = re.findall(r'\b\w*\s*(yes|no)\s*\w*\b', res_answer)
            elif num_class == 4:
                res = re.findall(r'\b\w*\s*(rather yes|rather no)\s*\w*\b', res_answer)
                if len(res) == 0:
                    res = re.findall(r'\b\w*\s*(yes|no)\s*\w*\b', res_answer)
            elif num_class == 5:
                res = re.findall(r'\b\w*\s*(rather yes|rather no)\s*\w*\b', res_answer)
                if len(res) == 0:
                    res = re.findall(r'\b\w*\s*(yes|no)\s*\w*\b', res_answer)
                if len(res) == 0:
                    res = re.findall(r'\b\w*\s*(neutral)\s*\w*\b', res_answer)
    
            res_answer = "None" if len(res) == 0 else res[0]

            match_map = {
                "None": None,
                "no": -2,
                "rather no": -1,
                "neutral": 0,
                "rather yes": 1,
                "yes": 2
            }
            res_answer = match_map[res_answer]
        elif answer_type == "number":
            res_answer = re.search(r'-?[0-9]\d*', res_answer)

            if res_answer:
                res_answer = int(res_answer.group())
            else:
                res_answer = None

        return res_answer

    @classmethod
    def postprocess_results(
        cls, 
        data, 
        answer_type="", 
        num_class=5,
    ):
        # Check if there is the column named "model"
        try:
            data["model"]
        except:
            print("There is no 'model' column in the dataset!")

        data["model_extracted"] = [cls.extract_answer(x, num_class, answer_type) for x in data["model"]]
        
        # Check if there are NaNs in the dataframe
        if np.isnan(data["model_extracted"].unique()).any():
            none_wordsets = data[np.isnan(data["model_extracted"])]["task"]
            print("The number of samples in the results dataset with 'None' extracted answer:", none_wordsets.shape[0])

            for wordset in none_wordsets:
                print("Wordset:", wordset)
                print("Model answer:", data[data["task"] == wordset]["model"].item())
                print("Extracted answer:", data[data["task"] == wordset]["model_extracted"].item())

            # Remove all samples with "NaN" exatracted answer from the results dataset
            data = data[[not elem for elem in np.isnan(data["model_extracted"])]]
            
        return data

    @classmethod
    def plot_answers_distrib(
            cls,
            data,
            model_name='',
            dataset_name='',
            fig_size=(8, 6),
            container_size=12,
            title_size=15,
            ticks_size=12,
            label_size=12,
            save_dir=None,
            print_plot=True,
            num_class=2,
            custom_mapper=num_to_words
            ):
        df = data[["true", "model_extracted"]]
        df.columns = ["Human", "Model"]

        if custom_mapper.__name__ == "num_to_words":
            custom_mapper = cls.num_to_words

        if not isinstance(df.iloc[0]["Human"], str):
            
            df["Human"] = df["Human"].map(custom_mapper)
        if not isinstance(df.iloc[0]["Model"], str):
            df["Model"] = df["Model"].map(custom_mapper)

        df_long = df.melt(var_name='Column', value_name='Response')

        if num_class == 5:
            category_order = ["bad", "rather_bad", "neutral", "rather_good", "good"]
        elif num_class == 4:
            category_order = ["bad", "rather_bad", "rather_good", "good"]
        elif num_class == 2:
            category_order = ["bad", "good"]

        df_long['Response'] = pd.Categorical(df_long['Response'], categories=category_order, ordered=True)

        palette_colors = {"Human": "#3A3875", "Model": "#D366D3"}

        plt.figure(figsize=fig_size)

        ax = sns.histplot(
            data=df_long,
            x='Response',
            hue='Column',
            multiple='dodge',
            shrink=0.8,
            palette=palette_colors,
        )

        for i in ax.containers:
            ax.bar_label(i, fontsize=container_size)

        plt.title(f'Distribution of Human and {model_name} answers. {dataset_name}', fontsize=title_size)

        plt.xticks(fontsize=ticks_size)
        plt.yticks(fontsize=ticks_size)

        ax.set_xlabel('Response', fontsize=label_size)
        ax.set_ylabel('Count', fontsize=label_size)

        if save_dir is not None:
            plt.savefig(save_dir)
            print(f"Plot was saved to the file {save_dir}")

        if print_plot:
            plt.show()
        else:
            plt.close()

    @classmethod
    def plot_conf_matrix(
            cls,
            true,
            predict,
            model_name="",
            heatmap_fontsize=10,
            axis_label_fontsize=13,
            title_fontsize=17,
            ticklabel_fontsize=20,
            save_dir=None,
            print_plot=False,
            num_class=2
        ):

        if isinstance(true[0], str):
            true = [cls.words_to_num(x) for x in true]
        if isinstance(predict[0], str):
            predict = [cls.words_to_num(x) for x in predict]

        cm = confusion_matrix(true, predict)

        class_labels = {
                -2: 'bad',
                -1: 'rather_bad',
                0: 'neutral',
                1: 'rather_good',
                2: 'good'}
        if num_class == 2:
            classes_num = [-2, 2]
        elif num_class == 4:
            classes_num = [-2, -1, 1, 2]
        elif num_class == 5:
            classes_num = [-2, -1, 0, 1, 2]
        sns.heatmap(cm,
                    annot=True,
                    fmt='g',
                    xticklabels=[class_labels[label] for label in classes_num],
                    yticklabels=[class_labels[label] for label in classes_num],
                    annot_kws={"fontsize": heatmap_fontsize}
         )

        plt.ylabel('Actual',fontsize=axis_label_fontsize)
        plt.xlabel('Prediction',fontsize=axis_label_fontsize)
        plt.title(f'Confusion Matrix {model_name}',fontsize=title_fontsize)

        plt.xticks(fontsize=ticklabel_fontsize)
        plt.yticks(fontsize=ticklabel_fontsize)

        if save_dir is not None:
            plt.savefig(save_dir)
            print(f"Plot was saved to the file {save_dir}")

        if print_plot:
            plt.show()
        else:
            plt.close()

    @classmethod
    def calculate_metrics(cls, data, save_dir=None, metric_flags=["f1-score", "accuracy", "phik", "lin_corr"]):
        if isinstance(data.iloc[0]["true"], str):
            data["true"] = data["true"].map(cls.words_to_num)
        if isinstance(data.iloc[0]["model_extracted"], str):
            data["model_extracted"] = data["model_extracted"].map(cls.words_to_num)
        
        phik_corr = "-"
        lin_corr = "-"
        accuracy = "-"
        f1_score_res = "-"

        if "phik" in metric_flags:
            # non-linear dependencies
            phik_corr = data[["true", "model_extracted"]].phik_matrix()
            phik_corr = phik_corr.loc["true"]["model_extracted"]
        if "lin_corr" in metric_flags:
            # linear_correlation
            lin_corr = data[["true", "model_extracted"]].corr()
            lin_corr = lin_corr.loc["true"]["model_extracted"]
        if "accuracy" in metric_flags:
            # accuracy
            accuracy = accuracy_score(data["true"], data["model_extracted"])
        if "f1-score" in metric_flags:
            # f1-score
            f1_score_res = f1_score(data["true"], data["model_extracted"], average='macro')

        metrics_df = pd.DataFrame(
            {
                "phik": [phik_corr],
                "pearson": [lin_corr],
                "accuracy": [accuracy],
                "f1_score": [f1_score_res]
            }
        )
        if save_dir is not None:
            metrics_df.to_csv(save_dir)

        return metrics_df
    
    @classmethod
    def classification_metric(self, data, num_class=2):
        if num_class == 2:
            metric_mapper = {
                -2: 0,
                2: 1
            }
        elif num_class == 4:
            metric_mapper = {
                -2: 0,
                -1: 1/3,
                1: 2/3,
                2: 1
            }
        elif num_class == 5:
            metric_mapper = {
                -2: 0,
                -1: 1/4,
                0: 2/4,
                1: 3/4,
                2: 1
            }

        return [metric_mapper[x] for x in data]

    def get_wordnet_pos(word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

        return tag_dict.get(tag, wordnet.NOUN)

    @classmethod
    def preprocess_topics(cls, data, data_preprocessor, lemmatizer):
        for i in range(len(data)):
            topics = data[i]

            for topic_id in range(len(topics)):
                topics[topic_id] = data_preprocessor.remove_punct(topics[topic_id])
                topics[topic_id] = data_preprocessor.remove_ext_spaces(topics[topic_id])
                topics[topic_id] = data_preprocessor.tokenize(topics[topic_id])
                topics[topic_id] = " ".join([lemmatizer.lemmatize(w, cls.get_wordnet_pos(w)) for w in topics[topic_id]])
            
            data[i] = topics
        return data
    
    @classmethod 
    def get_topic_metric(cls, data_true, data_pred):
        data_preprocessor = data_preprocessing.TextPreprocessor(lang='eng')
        lemmatizer = WordNetLemmatizer()
        model = SentenceTransformer('bert-base-nli-mean-tokens')

        data_pred = cls.preprocess_topics(data_pred, data_preprocessor, lemmatizer)
        data_true = cls.preprocess_topics(data_true, data_preprocessor, lemmatizer)

        # Calculate average cosine similarity 
        cos_sim_avg = []
        for i in tqdm(range(len(data_true))):
            sample_true = data_true[i]
            sample_pred = data_pred[i]

            topic_embeddings = model.encode(sample_pred)
            true_topic_embeddings = model.encode(sample_true)

            cos_sim = cosine_similarity(topic_embeddings, true_topic_embeddings)[0]
            cos_sim_avg.append(sum(cos_sim) / len(cos_sim))

        return cos_sim_avg




