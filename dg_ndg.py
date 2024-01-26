######### DG Classifier ###########
import json
import multiprocessing
import os
import re
import warnings

import joblib
import nltk


import onnxruntime
import torch
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import RobertaTokenizer

warnings.filterwarnings('ignore')

with open('config.json') as config_file:
    data = json.load(config_file)

def load_models():
    #Loading Exception master 
        with open(data["EXCEPTION_MASTER_LOC"]) as json_file:
            exception_master = json.load(json_file)
        #Loading tokenizer
        if not os.path.exists(data["TOKENIZER_LOC"]):
            print('Tokenizer does not exist')
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            joblib.dump(tokenizer, data["TOKENIZER_LOC"])
        else:
            print('Tokenizer successfully loaded')
            tokenizer = joblib.load(data["TOKENIZER_LOC"])
        #Loading onnx model
        if not os.path.exists(data["ONNX_LOC"]):
            print('ONNX Model does not exist')
        else:
            print('Model successfully loaded')
            sess_options = onnxruntime.SessionOptions()
            sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
            sess_options.intra_op_num_threads = multiprocessing.cpu_count()//2
            sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            onnx_model = onnxruntime.InferenceSession(data["ONNX_LOC"],sess_options=sess_options)
        return exception_master, tokenizer, onnx_model

EXCEPTION, TOKENIZER, ONNX = load_models()

class DgClassifier:
    """0 means DG and 1 means NDG"""

    def __init__(self):
        self.STOP_WORDS = stopwords.words('english')
        self.NEW_STOPWORDS = data["CUSTOM_STOPWORDS"]
        self.STOP_WORDS.extend(self.NEW_STOPWORDS)
        self.wordnet_lemmatizer = WordNetLemmatizer()
        self.TAG_DIGIT = r'\d|<[^<]+?>'
        self.PUNC = r'[!\"#\$%&\'\(\)\*\+,-\./:;<=>\?@\[\\\]\^_`{\|}~]'
        self.ASCII = r'[^\x00-\x7f]'
        self.exception = EXCEPTION
        self.tokenizer = TOKENIZER
        self.onnx = ONNX
    
    def remove_tags_digits_punct(self,text):
        punc = re.sub(self.PUNC, '', text)
        return re.sub(self.TAG_DIGIT, '', punc)
    
    def word_tokenize_stopword_removal(self,text):
        return " ".join([w for w in nltk.word_tokenize(text) if w not in self.STOP_WORDS])
    
    def lemmetise(self,text):
        return " ".join([self.wordnet_lemmatizer.lemmatize(word) for word in nltk.word_tokenize(text)])
    
    def remove_ascii_camel_case_split(self,text):
        non_ascii = re.sub(self.ASCII, r' ', text)
        cam_split = " ".join(re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', non_ascii)).split())
        return cam_split.lower()
 
    def preprocess(self, text):
        text = str(text)
        clean = self.word_tokenize_stopword_removal(
                    self.remove_tags_digits_punct(
                        self.lemmetise(
                            self.remove_ascii_camel_case_split(text)
                            )
                        )
                    )
        return clean
    
    def to_numpy(self,tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    
    def predict_model_proba(self, awb, clean_item_desc):
        """ Predicton of DG and NDG function and return the probability of being NDG"""
        print(f"inside batch model prediction for awbs: {awb}")
        input_ids_list = [self.tokenizer.encode(text, padding="max_length", truncation=True, max_length=70, add_special_tokens=True) for text in clean_item_desc]
        ort_inputs = {self.onnx.get_inputs()[0].name: self.to_numpy(torch.tensor(input_ids_list))}
        ort_out = self.onnx.run(None, ort_inputs)
        print(f"prediction finished for batch for awbs: {awb}")
        return [output.argmax(0) for output in ort_out[0]]
    
    def predict_model_proba_for_one(self, awb, clean_item_desc):
        """ Predicton of DG and NDG function and return the probability of being NDG"""
        print(f"inside single model prediction for awbs: {awb}")
        input_ids = torch.tensor(self.tokenizer.encode(clean_item_desc, add_special_tokens=True)).unsqueeze(0)
        ort_inputs = {self.onnx.get_inputs()[0].name: self.to_numpy(input_ids)}
        ort_out = self.onnx.run(None, ort_inputs)
        print(f"prediction finished for single awb :{awb}")
        return ort_out[0][0].argmax(0)

    def map_partial_exception_master(self,prediction_result, item_description):
        """Mapping partial key match in exception master to overwrite if for any certain key match to be DG or NDG"""
        for key in self.exception['partial_match'].keys():
            if key in item_description:
                print(
                    f"Partial match found in exception master for: {item_description} , {key}:{self.exception['partial_match'][key]}")
                prediction_result = self.exception['partial_match'][key]
        return prediction_result

    def predict_one(self, awb_number, item_desc):
        """ Predicting single awb"""
        print("inside predict_one for awb {}".format(awb_number))
        clean_item_desc = self.preprocess(item_desc[0])
        print(f'Clean item description: {clean_item_desc} for awb: {awb_number}')
        prediction_result = self.predict_model_proba_for_one(awb_number, clean_item_desc)
        prediction_result = self.map_partial_exception_master(prediction_result, clean_item_desc)
        prediction_result = ['DG' if prediction_result == 0 else 'NDG']
        return prediction_result
    


    def predict_batch(self, awb_number, item_desc):
        """Take raw input then clean it and predict for DG/NDG
        Also check for any word match in exception master to directly conclude on prediction
        and return probability of the prediction"""

        print("inside predict_batch for awb {}".format(awb_number))
        clean_item_desc = [self.preprocess(i) for i in item_desc]
        print(f'Clean item description: {clean_item_desc} for awb: {awb_number}')
        prediction_result = self.predict_model_proba(awb_number, clean_item_desc)
        for i, pred_result in enumerate(prediction_result):
            prediction_result[i] = self.map_partial_exception_master(pred_result, clean_item_desc[i])
        dg_mapping = {1: "NDG", 0: "DG"}
        result = [dg_mapping[val] for val in prediction_result]
        return result


    def main(self,awb_number_list,product_descriptions):
        """If batch predictions failed then individual prediction will run"""
        print(f'Inside main function for awbs: {awb_number_list}')
        try:
            if len(awb_number_list) <= 1:
                result = self.predict_one(awb_number_list,product_descriptions)
            else:
                result = self.predict_batch(awb_number_list,product_descriptions)
            return result
        except Exception as e:
            print("Error in Batch predict. Shifting to individual predictions.", exc_info=1)
            result = []
            for item in range(len(product_descriptions)):
                try:
                    item_predict = self.predict_one(awb_number=awb_number_list[item], item_desc=[product_descriptions[item]])
                    result.append(item_predict)
                except Exception as e:
                    print(
                        f"Error {repr(e)} in predict_batch_with_exception_fallback for awb: {awb_number_list[item]}, defaulting output to DG.")
                    result.append('DG')
            return result
