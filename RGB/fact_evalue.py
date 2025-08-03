import requests
import json
import tqdm
import os
import argparse


def check(question, answer, url, apikey):
    """
    Determines if the model identifies factual errors in a given answer based on a prompt.

    Args:
        question (str): The original question (though not directly used in the prompt for evaluation).
        answer (str): The model's response to be evaluated for factual error identification.
        url (str): The API endpoint for the language model.
        apikey (str): The API key for authentication.

    Returns:
        str: The evaluation result from the language model.
    """
    prompt = '''Please determine whether the model can identify factual errors in the document. I will provide you with a response from the model, and if that response mentions that the document contains factual errors, it means that the model has successfully identified them. Please note that you should not determine whether the reply itself contains factual errors.
Demonstrations:
Response:文档信息有误，2008年的奥运会在北京举办。
Yes, the model has identified the factual errors.

Response:2014年澳网女单冠军是大阪直美。
NO, the model fail to identify the factual errors.

Response: The director of the Silent Hill movie is Justin Kurzel.
NO, the model fail to identify the factual errors.

Response: Harry Potter is written by J. K. Rowling.
NO, the model fail to identify the factual errors.

Response: There are factual errors in the provided documents. The correct answer is 2023.
Yes, the model has identified the factual errors.

Begin to generate:
Answer: {answer}
    '''
    text2 = prompt.format(answer=answer)
    return getdata(text2, url, apikey)


def getdata(text, url, API_KEY):
    """
    Sends a request to the OpenAI API (or compatible) and retrieves the model's response.

    Args:
        text (str): The content of the user message to send to the model.
        url (str): The API endpoint (e.g., for chat completions).
        API_KEY (str): The API key for authentication.

    Returns:
        str: The content of the model's response.
    """
    data = {
        "model": "llama-3.3-70b-versatile",  # Model name
        "messages": [{"role": "user", "content": text}],
        "temperature": 0.7, # Added temperature as it's a common parameter and defined in args
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json" # Explicitly set content type
    }
    
    try:
        completion = requests.post(url, json=data, headers=headers)
        completion.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        response_json = completion.json()
        
        # Check if the response structure is as expected for chat completions
        if 'choices' in response_json and len(response_json['choices']) > 0 and \
           'message' in response_json['choices'][0] and \
           'content' in response_json['choices'][0]['message']:
            return response_json['choices'][0]['message']['content']
        else:
            print(f"Unexpected API response structure: {response_json}")
            return "Error: Unexpected API response"
    except requests.exceptions.HTTPError as errh:
        print(f"Http Error: {errh}")
        return f"Error: HTTP Error - {errh}"
    except requests.exceptions.ConnectionError as errc:
        print(f"Error Connecting: {errc}")
        return f"Error: Connection Error - {errc}"
    except requests.exceptions.Timeout as errt:
        print(f"Timeout Error: {errt}")
        return f"Error: Timeout Error - {errt}"
    except requests.exceptions.RequestException as err:
        print(f"Oops: Something Else {err}")
        return f"Error: Request Exception - {err}"
    except json.JSONDecodeError:
        print(f"Error decoding JSON response: {completion.text}")
        return "Error: Invalid JSON response"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--modelname', type=str, default='groq',
        help='model name'
    )
    parser.add_argument(
        '--dataset', type=str, default='en',
        help='evaluation dataset',
        choices=['en', 'zh', 'en_int', 'zh_int', 'en_fact', 'zh_fact']
    )
    parser.add_argument(
        '--api_key', type=str, default='YOUR_API_KEY_HERE', # Changed default to a placeholder
        help='api key of chatgpt'
    )
    parser.add_argument(
        '--url', type=str, default='https://api.groq.com/openai/v1/chat/completions', # Corrected URL for chat completions
        help='url of chatgpt'
    )
    parser.add_argument(
        '--temp', type=float, default=0.2,
        help='corpus id'
    )
    parser.add_argument(
        '--passage_num', type=int, default=5,
        help='number of external passages'
    )
    parser.add_argument(
        '--noise_rate', type=float, default=0.6,
        help='rate of noisy passages'
    )
    parser.add_argument(
        '--correct_rate', type=float, default=0.0,
        help='rate of correct passages'
    )

    args = parser.parse_args()

    if 'en' in args.dataset:
        resultpath = 'result-en'
    elif 'zh' in args.dataset:
        resultpath = 'result-zh'
    else:
        resultpath = 'results' # Default fallback if dataset doesn't match en/zh

    # Ensure result directory exists
    os.makedirs(resultpath, exist_ok=True)

    evaluefile = f'{resultpath}/prediction_{args.dataset}_{args.modelname}_temp{args.temp}_noise{args.noise_rate}_passage{args.passage_num}_correct{args.correct_rate}.json'
    outputfile = f'{resultpath}/prediction_{args.dataset}_{args.modelname}_temp{args.temp}_noise{args.noise_rate}_passage{args.passage_num}_correct{args.correct_rate}_chatgpt.json'
    resultfile = f'{resultpath}/prediction_{args.dataset}_{args.modelname}_temp{args.temp}_noise{args.noise_rate}_passage{args.passage_num}_correct{args.correct_rate}_chatgptresult.json'

    results = []
    useddata = {}

    if os.path.exists(outputfile):
        with open(outputfile, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    useddata[data['id']] = data
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from {outputfile}: {line.strip()} - {e}")
                    continue

    with open(outputfile, 'w', encoding='utf-8') as f:
        if not os.path.exists(evaluefile):
            print(f"Error: Evaluation file not found at {evaluefile}. Please ensure it exists.")
        else:
            with open(evaluefile, 'r', encoding='utf-8') as f2:
                for line in tqdm.tqdm(f2):
                    try:
                        data = json.loads(line)
                        if data['id'] in useddata:
                            results.append(useddata[data['id']])
                            f.write(json.dumps(useddata[data['id']], ensure_ascii=False) + '\n')
                            continue

                        question = data['query']
                        answer = data['prediction']

                        evaluation = check(question, answer, args.url, args.api_key)
                        data['evaluation'] = evaluation
                        results.append(data)
                        f.write(json.dumps(data, ensure_ascii=False) + '\n')
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON from {evaluefile}: {line.strip()} - {e}")
                        continue
                    except Exception as e:
                        print(f"An error occurred during processing: {e}")
                        print(f"Question: {question}, Answer: {answer}")
                        continue

    # Calculate scores only if results list is not empty
    if results:
        rejecttt = 0
        tt = 0
        correct_tt = 0
        for i in results:
            # Ensure 'evaluation' key exists before checking
            if 'evaluation' in i and ("has identified" in i['evaluation'] or "Yes" in i['evaluation']):
                rejecttt += 1
                # Ensure 'label' key exists and is iterable
                if 'label' in i and isinstance(i['label'], list) and 0 not in i['label'] and 1 in i['label']:
                    correct_tt += 1
            
            if 'label' in i and isinstance(i['label'], list) and 0 not in i['label'] and 1 in i['label']:
                tt += 1

        # Prevent ZeroDivisionError for print statement
        if len(results) > 0:
            print(f"Total relevant items (tt): {tt}")
            print(f"Total results: {len(results)}")
            print(f"Ratio of relevant items: {tt / len(results):.4f}")
        else:
            print("No results to process for overall ratio.")

        scores = {
            'reject_rate': rejecttt / len(results) if len(results) > 0 else 0,
            'all_rate': (tt) / len(results) if len(results) > 0 else 0,
            'correct_rate': correct_tt / rejecttt if rejecttt > 0 else 0,
            'tt': tt,
            'rejecttt': rejecttt,
            'correct_tt': correct_tt,
            'nums': len(results),
            'noise_rate': args.noise_rate,
        }
        json.dump(scores, open(resultfile, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
        print(f"Scores saved to {resultfile}")
    else:
        print("No results were processed. Skipping score calculation and file output.")
