FROM amazon/aws-lambda-python:3.8

COPY dg_ndg.py config.json exception_master.json lambda_function.py roberta_tokenizer.pkl roberta.onnx requirements.txt ./

RUN python3.8 -m pip install --upgrade pip

RUN python3.8 -m pip --default-timeout=100 install --no-cache-dir -r requirements.txt

RUN python3.8 -m nltk.downloader punkt

RUN python3.8 -m nltk.downloader stopwords

RUN python3.8 -m nltk.downloader wordnet

CMD ["lambda_function.lambda_handler"]