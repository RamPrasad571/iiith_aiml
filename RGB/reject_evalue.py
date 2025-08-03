import requests
import json
import tqdm, os
import argparse

def check(question, answer, url, apikey):
    """
    Constructs a prompt for a language model to evaluate if an answer
    addresses a given question based on retrieved documents.
    """
    prompt = '''I will give you a question and an answer generated through document retrieval. Please use this answer to determine if the retrieved document can solve the question.
Demonstrations:
Question: 2023年澳网女单冠军是谁
Answer:文档信息不足，因此我无法基于提供的文档回答该问题。
No, the question is not addressed by the documents.

Question: Who is the champion of Australian Open 2023 Women's Singles?
Answer: Serena Williams
Yes, the question is addressed by the documents.

Question: Where is ACL2023 held?
Answer: Location of ACL2023 has not been confirmed.
No, the question is not addressed by the documents.

Question: 2023年中国GDP是多少?
Answer: I can not answer this question。
No, the question is not addressed by the documents.

Begin to generate:
Question: {question}
Answer: {answer}
    '''
    text2 = prompt.format(question=question, answer=answer)
    return getdata(text2, url, apikey)


def getdata(text, url, API_KEY):
    """
    Sends a request to the OpenAI API (or compatible endpoint) to get a completion.
    """
    data = {
        "model": "llama-3.3-70b-versatile",  # Model name
        "messages": [{"role": "user", "content": text}]
    }
    headers = {"Authorization": f"Bearer {API_KEY}"}
    
    try:
        completion = requests.post(url, json=data, headers=headers)
        completion.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        response_json = completion.json()
        
        # Check if 'choices' and its elements exist before accessing
        if 'choices' in response_json and len(response_json['choices']) > 0 and \
           'message' in response_json['choices'][0] and \
           'content' in response_json['choices'][0]['message']:
            return response_json['choices'][0]['message']['content']
        else:
            print(f"Unexpected API response structure: {response_json}")
            return "Error: Unexpected API response"
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return f"Error: Request failed - {e}"
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}, Response content: {completion.text}")
        return f"Error: JSON decode error - {e}"
    except KeyError as e:
        print(f"KeyError in API response: {e}, Response: {response_json}")
        return f"Error: KeyError in API response - {e}"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--modelname', type=str, default='groq',
        help='model name'
    )
    parser.add_argument(
        '--dataset', type=str, default='en',
        help='evaluation dataset',
        choices=['en','zh','en_int','zh_int','en_fact','zh_fact']
    )
    parser.add_argument(
        '--api_key', type=str, default='api_key',
        help='api key of chatgpt'
    )
    parser.add_argument(
        '--url', type=str, default='https://api.groq.com/openai/v1/chat/completions', # Changed to chat completions endpoint
        help='url of chatgpt'
    )
    parser.add_argument(
        '--temp', type=float, default=0.7,
        help='temperature for generation' # Changed help text for clarity
    )
    parser.add_argument(
        '--passage_num', type=int, default=5,
        help='number of external passages'
    )

    args = parser.parse_args()

    if 'en' in args.dataset:
        resultpath = 'result-en'
    elif 'zh' in args.dataset:
        resultpath = 'result-zh'
    else:
        # Default path if dataset choice is not 'en' or 'zh' based
        resultpath = 'result-other' 
    
    # Ensure resultpath directory exists
    os.makedirs(resultpath, exist_ok=True)

    evaluefile = f'{resultpath}/prediction_{args.dataset}_{args.modelname}_temp{args.temp}_noise{1.0}_passage{args.passage_num}_correct{0.0}.json'
    outputfile = f'{resultpath}/prediction_{args.dataset}_{args.modelname}_temp{args.temp}_noise{1.0}_passage{args.passage_num}_correct{0.0}_chatgpt.json'
    resultfile = f'{resultpath}/prediction_{args.dataset}_{args.modelname}_temp{args.temp}_noise{1.0}_passage{args.passage_num}_correct{0.0}_chatgptresult.json'

    results = []
    useddata = {}
    
    # Load existing data to avoid re-processing
    if os.path.exists(outputfile):
        print(f"Loading existing data from {outputfile}...")
        with open(outputfile, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    useddata[data['id']] = data
                except json.JSONDecodeError as e:
                    print(f"Skipping malformed line in {outputfile}: {line.strip()} - Error: {e}")
        print(f"Loaded {len(useddata)} existing records.")

    print(f"Processing evaluation file: {evaluefile}")
    if not os.path.exists(evaluefile):
        print(f"Error: Evaluation file '{evaluefile}' not found. Please ensure it exists and contains valid JSON lines.")
        exit() # Exit if the input file doesn't exist

    processed_count = 0
    with open(outputfile, 'a', encoding='utf-8') as f_out: # Changed to 'a' (append mode) to avoid overwriting
        with open(evaluefile, 'r', encoding='utf-8') as f_in:
            for line in tqdm.tqdm(f_in, desc="Processing data"):
                try:
                    data = json.loads(line)
                    
                    # Check if this data point has already been processed correctly
                    if data['id'] in useddata and \
                       data.get('query') == useddata[data['id']].get('query') and \
                       data.get('prediction') == useddata[data['id']].get('prediction') and \
                       'evaluation' in useddata[data['id']]: # Ensure 'evaluation' key exists
                        
                        results.append(useddata[data['id']])
                        f_out.write(json.dumps(useddata[data['id']], ensure_ascii=False) + '\n')
                        processed_count += 1
                        continue

                    question = data.get('query', '') # Use .get() for safer access
                    answer = data.get('prediction', '') # Use .get() for safer access
                    
                    if not question or not answer:
                        print(f"Skipping record with missing 'query' or 'prediction': {data}")
                        continue

                    evaluation = check(question, answer, args.url, args.api_key)
                    data['evaluation'] = evaluation
                    results.append(data)
                    f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                    processed_count += 1

                except json.JSONDecodeError as e:
                    print(f"Skipping malformed line in {evaluefile}: {line.strip()} - Error: {e}")
                    continue
                except Exception as e:
                    print(f"An error occurred during processing: {e}")
                    print(f"Problematic data: Question: '{question}', Answer: '{answer}'")
                    # Do not continue here, let it try to process the next line
                    # If this error is persistent, the user needs to fix the API key/URL or model
                    continue
    
    print(f"\nFinished processing. Total records processed and added to results: {processed_count}")
    print(f"Total records in 'results' list: {len(results)}")

    if len(results) == 0:
        print("\nError: The 'results' list is empty. Cannot calculate scores due to ZeroDivisionError.")
        print("Possible reasons:")
        print("1. The input file (evaluefile) was empty or contained no valid JSON lines.")
        print("2. All API calls failed, or an unexpected error occurred for every record.")
        print("3. The conditions for reusing 'useddata' were not met, and new processing failed.")
        print("Please check the console output for specific error messages during processing.")
        exit() # Exit to prevent ZeroDivisionError

    rejecttt = 0
    tt = 0
    for i in results:
        # Ensure 'evaluation' key exists before checking
        if "not addressed" in i.get('evaluation', ''): 
            rejecttt += 1
        # Ensure 'label' key exists and is a list before checking
        if isinstance(i.get('label'), list) and 0 not in i['label'] and 1 in i['label']:
            tt += 1
    
    # Calculate scores only if results list is not empty
    reject_rate = rejecttt / len(results)
    all_rate = tt / len(results)

    print(f"True Positive Rate (tt/len(results)): {all_rate}")
    
    scores = {
        'reject_rate': reject_rate,
        'all_rate': all_rate,
        'tt': tt,
        'rejecttt': rejecttt,
        'nums': len(results),
    }
    
    print(f"\nSaving final results to {resultfile}")
    json.dump(scores, open(resultfile, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    print("Script finished successfully.")

