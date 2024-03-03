import argparse
from ast import parse
import json
import os
import time
import openai
from tqdm import tqdm
import google.generativeai as genai
import jsonlines

# import shortuuid
import asyncio
from typing import Any
import logging


model = genai.GenerativeModel('gemini-pro')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
async def dispatch_openai_requests(
    messages_list: list[list[dict[str,Any]]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
) -> list[str]:
    """
    Dispatches requests to OpenAI API asynchronously.
    
    Args:
        messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        model: OpenAI model to use.
        temperature: Temperature to use for the model.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use for the model.

    Returns:
        List of responses from OpenAI API.
    """
    async_responses = [
        openai.ChatCompletion.acreate(
            model=model,
            messages=x,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)

def parse_score(review):
    try:
        score = float(review.split('\n')[0])
    except Exception as e:
        if('pont:' in review):
            score = float(review.split('pont:')[1].split()[0])
        elif('Pont:' in review):
            score = float(review.split('Pont:')[1].strip()[0])
        if('pontszám:' in review):
            score = float(review.split('pontszám:')[1].split()[0])
        elif('Pontszám:' in review):
            score = float(review.split('Pontszám:')[1].strip()[0])
        else:           
            logger.error(
                f"{e}\nContent: {review}\n" "You must manually fix the score pair."
            )
            score = -1
    
    return score

def find_error_items(alpaca_data,alpaca_data_cleaned_archive):
    alpaca_data_cleaned_archive_str = set([str(d) for d in alpaca_data_cleaned_archive])
    dirty_list = []
    for i, x in enumerate(alpaca_data):
        x = str(x)
        if(x not in alpaca_data_cleaned_archive_str):
            dirty_list.append(i)
    return dirty_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChatGPT-based QA evaluation.")
    parser.add_argument("-o", "--output-review-file")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="maximum number of tokens produced in the output",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="the batch size to call the ChatGPT."
    )
    args = parser.parse_args()
    # alpaca_data_cleaned_archive = json.load(open("./alpaca_data_cleaned_archive.json"))
    # alpaca_data = json.load(open("./alpaca_data.json"))
    data = []
    with jsonlines.open('alpaca_hu_v2.jsonl') as reader:
        for obj in reader:
            data.append(obj)

    from datasets import load_dataset
    dataset = load_dataset("Bazsalanszky/budapest-v0.1-hun")
    data = list(dataset['train'])

    data_set_ids = set([])
    # with jsonlines.open('alpaca_hu_v2_ratings.jsonl') as reader:
    #     for obj in reader:
    #         data_set_ids.remove(obj['idx'])

    # system_prompt = "We would like to request your feedback on the performance of AI assistant in response to the instruction and the given input displayed following."
    system_prompt = "Szeretnénk a segítségedet kérni, egy MI asszisztens értékelésével kapcsolatban, amely egy utasításra ad választ. Kritikus értékelésekre számítunk."


    '''
    rating according to the helpfulness
    '''
    # user_prompt = "Please rate according to the helpfulness of the response to the instruction and the input. Each assistant receives an score on a scale of 0 to 5, where a higher score indicates higher level of the helpfulness. Please first output a single line containing value indicating the scores. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias. \n\n"
    # dirty_list = find_error_items(alpaca_data, alpaca_data_cleaned_archive)
    '''
    rating according to the accuracy
    '''
    # user_prompt = "Please rate according to the accuracy of the response to the instruction and the input. Each assistant receives a score on a scale of 0 to 5, where a higher score indicates higher level of the accuracy. Please first output a single line containing value indicating the scores. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias. \n\n"
    user_prompt = "Kérlek értékeld a bemenetre adott választ helyesség szerint. Adj egy pontszámot 0 és 10 között, ahol a magasabb pontszám nagyobb pontosságot és jobb nyelvhelyességet jelent. Kérlek előbb írj egy bekezdést, melyben szövegesen értékeled a megoldást. Ezt követően új sorban add meg a pontszámot. \n\n"
    print(f"Alpaca data pairs: {len(data)}")

    predictions = []
    i = 0
    wait_base = 10
    retry = 0
    batch_size = args.batch_size
    pbar = tqdm(total=len(data))
    with jsonlines.open(args.output_review_file, mode='a') as writer:
        while(i<len(data)):
            if retry > 2:
                i += 1
                pbar.update(1)
                retry = 0
            # if i not in data_set_ids:
            #     i += 1
            #     pbar.update(1)
            #     retry = 0
            #     continue
            try:
                triplet = "### Utasítás:\n{instruction}\n\n### Válasz:\n{output}\n\n".format_map(data[i])
                eval_prompt = triplet + user_prompt

                gprompt = system_prompt + '\n\n' + triplet + user_prompt
                gmessage =[
                            {
                                "role": "user",
                                "parts": [eval_prompt],
                            },
                ]
                response = model.generate_content(gmessage, 
                                    generation_config=genai.types.GenerationConfig(
                                        max_output_tokens=args.max_tokens,
                                        temperature=0.0,
                                        candidate_count=1,
                                        ))

                wait_base = 10
                pbar.update(batch_size)
                meta_output = {
                    'idx': i,
                    "input": data[i]['instruction'],
                    "output": data[i]['output'],
                    "triplet":triplet,
                    "review": response.text.strip(),
                }
                writer.write(meta_output)
                i += batch_size
                retry = 0

            except Exception as e:
                retry += 1
                print(e)
                print("Batch error: ",i, i+10)
                print("retry number: ", retry)
                time.sleep(wait_base)
                wait_base = wait_base*2
    pbar.close()
# if __name__=="__main__":
#     review = "3.5\n\nThe response provides a basic understanding of the difference between individual and societal performance. However, it lacks depth and does not provide specific examples or analysis to support the comparison and contrast. The language used is clear and concise, but the response could benefit from more elaboration and explanation. Overall, the response is helpful to a certain extent, but could be improved with more detail and analysis."
#     print(parse_score(review))
