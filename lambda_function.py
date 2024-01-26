import base64
import json
import time
from typing import Any, Dict, List

from dg_ndg import *


def validate_request(request):
    awb_numbers, product_descriptions = [],[]
    for item in request:
        awb_numbers.append(item['awb_number'])
        product_descriptions.append(item['product_description'])
    return awb_numbers, product_descriptions

 

def post_process(awb_numbers, prediction):
    dg_list, ndg_list = [], []
    for i in range(len(awb_numbers)):
        if prediction[i] == 'DG':
            dg_list.append(awb_numbers[i])
        else:
            ndg_list.append(awb_numbers[i])
    return {'ndg_list': ndg_list, 'dg_list': dg_list}


def lambda_handler(event, context):
    start_time = time.perf_counter()
    print('Request Received at:',start_time)
    print('Event is: ',event)
    file_content = base64.b64decode(event['content']) if 'content' in event else event.get('body', {})

    try:
        json_request = file_content if isinstance(file_content, dict) else json.loads(file_content)
        print('Json Request is: ',json_request)
        awb_numbers, product_descriptions = validate_request(json_request)
        dg_ndg = DgClassifier()
        processed_data = dg_ndg.main(awb_numbers,product_descriptions)
        response = post_process(awb_numbers,processed_data) 
        print('Returning Response at:',time.perf_counter() - start_time)
        http_res = {}
        http_res['statusCode'] = 200
        http_res['headers'] = {}
        http_res['headers']['Content-Type'] = 'application/json'
        http_res['body'] = json.dumps(response)
        return http_res    
    except Exception as e:
        print(e)
        response = {"ndg_list":[],"dg_list":[]}
        http_res = {}
        http_res['statusCode'] = 200
        http_res['headers'] = {}
        http_res['headers']['Content-Type'] = 'application/json'
        http_res['body'] = json.dumps(response)
        return http_res