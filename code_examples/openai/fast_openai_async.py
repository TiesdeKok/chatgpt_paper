# imports
import os, re, time, json, copy, logging, random
import aiohttp  
import argparse 
import asyncio  
import tiktoken 
from dataclasses import dataclass
import pandas as pd
import numpy as np
import datetime
import pathlib
import more_itertools as mit

def load_jsonl(file, to_pandas = False):

    with open(file, "r") as f:
        ret_list = []
        for item in f.readlines():
            ret_list.append(json.loads(item))
            
        if to_pandas:
            ret_df = pd.DataFrame(ret_list)
            return ret_df
        else: 
            return ret_list
        
def calc_tokens_used(done_file, verbose = 0):
    raw_data = load_jsonl(done_file)

    usage_tracker = []
    for item in raw_data:
        usage = item[1]["usage"]
        usage["qanda_id"] = item[-1]
        if verbose > 0:
            print(item[-1], usage["total_tokens"])
        usage_tracker.append(usage)

    usage_df = pd.DataFrame(usage_tracker)

    total_prompt = usage_df.prompt_tokens.sum()
    total_completion  = usage_df.completion_tokens.sum()
    
    ## IMPORTANT: THESE ARE HARD-CODED VALUES THAT YOU MIGHT NEED TO ADJUST
    
    cost_dict = {
        "gpt-4" : {"input" : 0.03, "output" : 0.06},
        "gpt-35-4k" : {"input" : 0.0015, "output" : 0.002},
        "gpt-35-16k" : {"input" : 0.003, "output" : 0.004},
        "gpt-35-ft" : {"input" : 0.012, "output" : 0.016}
        
    }
    
    print("Cost estimates:\n")
    for model, costs in cost_dict.items():
        cost_estimate = (total_prompt / 1000) * costs["input"] + (total_completion / 1000) * costs["output"]
        print(f"- {model} - ${cost_estimate:.2f}")
    print()
    
    total_tokens = usage_df.total_tokens.sum()
    print(f"Total number of tokens used: {total_tokens:,}")

    display(usage_df.describe())
        
def proc_results(save_filepath, model_type, verbose = 0, json_fixer = None, end = None):
    results = load_jsonl(save_filepath)
    
    processed_results = []
    error_list = []
    for item in results:
        try:
            if model_type == "chat":
                raw_item = item[1]["choices"][0]["message"]["content"]
                if end:
                    raw_item = raw_item.replace(end, "").strip()
                try:
                    item_data = json.loads(raw_item)
                except Exception as e:
                    will_raise = True
                    if json_fixer:
                        try:
                            fixed_item = json_fixer(raw_item)
                            item_data = json.loads(fixed_item)
                            will_raise = False
                        except Exception as e:
                            raise e
                    else:
                        raise e
                        
                item_data["id"] = item[2]
                processed_results.append(item_data)
            else:
                ## Check if errors
                if isinstance(item[1], list):
                    error_list += item[1]
                else:            
                    item_id = item[2]
                    is_batch = item_id[:8] == "batch-=-"
                    if not is_batch:
                        item_data = {"id" : item[2]}
                        completion_data = item[1]["choices"][0]
                        item_data["text"] = completion_data["text"].strip()
                        conf = np.exp(completion_data["logprobs"]["token_logprobs"][0])
                        item_data["confidence"] = conf
                        processed_results.append(item_data)
                    else:
                        batch_id_lookup = {i: x for i, x in enumerate(item_id[8:].split("|"))}

                        ## This is a quick fix, as I accidentally ran some with n > 1
                        choices = item[1]["choices"]
                        if "n" in item[0].keys():
                            n = item[0]["n"]
                            choices, counter = [], 0
                            for ele in item[1]["choices"]:
                                if (ele["index"] % n) == 0:
                                    ele["index"] = counter
                                    choices.append(ele)
                                    counter += 1

                        for completion_data in choices:
                            item_data = {"id" : batch_id_lookup[completion_data["index"]]} 
                            item_data["text"] = completion_data["text"].strip()
                            conf = np.exp(completion_data["logprobs"]["token_logprobs"][0])
                            item_data["confidence"] = conf
                            processed_results.append(item_data)

        except Exception as e:
            if verbose > 0:
                print(e)
            error_list.append(item)
            
    return processed_results, error_list


async def make_batch_predictions(
    input_list,
    model,
    tmp_store_loc,
    result_store_loc,
    return_json = False,
    system_message = None,
    label = None,
    print_interval = 1,
    api_key = "auto",
    max_requests_per_minute = "auto",
    max_tokens_per_minute = "auto",
    max_attempts = 3,
    logging_level = logging.INFO,
    temperature = 0,
    end = None,
    max_tokens = None,
    logprobs = None,
    request_url = "auto", 
    prompt_template = None,
    store_prompt_loc = None,
    logit_bias = None,
    batch_size = 1, ## Only available for completion API
    json_fixer = None
):
    
    ## System message
    default_system_message = "You are a helpful assistant."
    include_system_message = False
    if system_message:
        include_system_message = True
    
    ## Get current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    
    ## Set model type
    model_type = "completion"
    if "gpt-4" in model:
        model_type = "chat"
    elif "gpt-3.5" in model:
        model_type = "chat"
        
    if request_url == "auto":
        request_url = "https://api.openai.com/v1/chat/completions"
        if model_type == "completion":
            request_url = "https://api.openai.com/v1/completions"
    
    ## IMPORTANT: THESE ARE HARD-CODED VALUES THAT YOU MIGHT NEED TO ADJUST
    
    ## Get params
    if max_requests_per_minute == "auto":
        if "gpt-4" in model:
            max_requests_per_minute = 200
        elif "gpt-3.5" in model:
            max_requests_per_minute = 3_500
        else: ## For GPT-3 models
            max_requests_per_minute = 3_000
                        
    if max_tokens_per_minute == "auto":
        if "gpt-4" in model:
            max_tokens_per_minute = 40_000
        elif model == "gpt-3.5-turbo-16k":
            max_tokens_per_minute = 180_000
        elif "gpt-3.5" in model:
            max_tokens_per_minute = 90_000
        else:
            max_tokens_per_minute = 250_000
            ## Multipliers do not appear to be in effect...
            if "curie" in model:
                max_tokens_per_minute = max_tokens_per_minute  #* 25
            elif "babbage" in model:
                max_tokens_per_minute = max_tokens_per_minute * 2 #* 100
            elif "ada" in model:
                max_tokens_per_minute = max_tokens_per_minute  #* 200
                
    #### To avoid hitting it, set to slighlty lower
    max_requests_per_minute = int(max_requests_per_minute * 0.97) 
    max_tokens_per_minute = int(max_tokens_per_minute * 0.88)
    
    print(f"Running with the following rate limits: {max_requests_per_minute:,} RPM and {max_tokens_per_minute:,} TPM")
            
    if api_key == "auto":
        api_key = os.environ["OPENAI_API_KEY"]
            
    ## Get token encoding name
    
    try:
        token_encoding_name = tiktoken.encoding_for_model(model).name
    except:
        if model_type == "chat":
            token_encoding_name = "cl100k_base"
        else:
            token_encoding_name = "r50k_base"
        
    ## Define files
    
    if not isinstance(tmp_store_loc, pathlib.Path):
        tmp_store_loc = Path(tmp_store_loc)
    
    if not isinstance(result_store_loc, pathlib.Path):
        result_store_loc = Path(result_store_loc)
    
    tmp_fn = f"jobs_{timestamp}.jsonl"
    save_fn = f"results_{timestamp}.jsonl"
    
    if label:
        tmp_fn = f"jobs_{timestamp}_{label}.jsonl"
        save_fn = f"results_{timestamp}_{label}.jsonl"
    
    requests_filepath = (tmp_store_loc / tmp_fn).as_posix()
    save_filepath = (result_store_loc / save_fn).as_posix()
        
    ## Create batch input, if requested
    
    if batch_size > 1:
        new_input_list = []
        batch_list = list(mit.chunked(input_list, batch_size))
        
        for batch_item in batch_list:
            prompt_list = [x["prompt"] for x in batch_item]
            id_list = [x["id"] for x in batch_item]
            batch_id = "batch-=-" + "|".join(id_list)
            new_input_list.append({
                "prompt" : prompt_list,
                "id" : batch_id
            })
            
        size_before, size_after = len(input_list), len(new_input_list)
        input_list = new_input_list
        
        print(f"Batching activated with size {batch_size}, which collapses the {size_before:,} items to {size_after:,} batches.")
    
    ## Create files
    
    ###
    
    jobs = []
    for item in input_list:
        if model_type == "chat":
            if include_system_message:
                req_obj = {
                    "model": model,
                    "temperature" : temperature,
                    "messages" : [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": item["prompt"]}
                    ],
                    "extra_id" : item["id"]
                }
            else:
                req_obj = {
                    "model": model,
                    "temperature" : temperature,
                    "messages" : [
                        {"role": "user", "content": item["prompt"]}
                    ],
                    "extra_id" : item["id"]
                }                
            
            if max_tokens:
                req_obj["max_tokens"] = max_tokens
        else:    
            req_obj = {
                "model": model,
                "temperature" : temperature,
                "prompt" : item["prompt"],
                "extra_id" : item["id"]
            }
            
            if logit_bias:
                req_obj["logit_bias"] = logit_bias
            
            if end:
                if isinstance(end, list):
                    req_obj["stop"] = end
                else:
                    req_obj["stop"] = [end]
            
            if max_tokens:
                req_obj["max_tokens"] = max_tokens
                
            if logprobs:
                req_obj["logprobs"] = logprobs
        
        jobs.append(req_obj)
        
    ## Store jobs
    
    with open(requests_filepath, "w") as f:
        for job in jobs:
            json_string = json.dumps(job)
            f.write(json_string + "\n")
            
    ## Store prompt and other metadata
    
    if store_prompt_loc:
        if not isinstance(store_prompt_loc, pathlib.Path):
            store_prompt_loc = Path(store_prompt_loc)
            
        prompt_fn = f"prompt_{timestamp}.json"
        if label:
            prompt_fn = f"prompt_{timestamp}_{label}.json"
            
        ## Get example prompts
            
        meta_dict = {
            "timestamp" : timestamp,
            "label" : label,
            "model" : model,
            "num_obs" : len(jobs),
            "system_message" : system_message,
            "prompt_template" : prompt_template,
            "example_prompt" : random.choice(input_list)["prompt"],
        }
            
        with open(store_prompt_loc / prompt_fn, "w", encoding = "utf-8") as f:
            json.dump(meta_dict, f)
            
    ## Submit requests
    
    print(f"Submitting {len(jobs)} jobs for {model}!")
    
    print(f"Results are saved to the following file: {save_filepath}")
            
    async def process_requests():
        await process_api_requests_from_file(
            requests_filepath,
            save_filepath,
            request_url,
            api_key,
            max_requests_per_minute,
            max_tokens_per_minute,
            token_encoding_name,
            max_attempts,
            logging_level,
            print_interval
        )
        
    await process_requests()
    
    print(f"Done, the results are available in the following file:\n{save_filepath}")
    
    if return_json:
        processed_results, error_list = proc_results(save_filepath, model_type, json_fixer = json_fixer)
        
        return processed_results, error_list
    

async def process_api_requests_from_file(
    requests_filepath: str,
    save_filepath: str,
    request_url: str,
    api_key: str,
    max_requests_per_minute: float,
    max_tokens_per_minute: float,
    token_encoding_name: str,
    max_attempts: int,
    logging_level: int,
    print_interval: int
):
    """Processes API requests in parallel, throttling to stay under rate limits."""
    # constants
    seconds_to_pause_after_rate_limit_error = 15
    seconds_to_sleep_each_loop = 0.001  # 1 ms limits max throughput to 1,000 requests per second

    # initialize logging
    logging.basicConfig(level=logging_level)
    logging.debug(f"Logging initialized at level {logging_level}")

    # infer API endpoint and construct request header
    api_endpoint = api_endpoint_from_url(request_url)
    request_header = {"Authorization": f"Bearer {api_key}"}

    # initialize trackers
    queue_of_requests_to_retry = asyncio.Queue()
    task_id_generator = task_id_generator_function()  # generates integer IDs of 1, 2, 3, ...
    status_tracker = StatusTracker()  # single instance to track a collection of variables
    next_request = None  # variable to hold the next request to call

    # initialize available capacity counts
    available_request_capacity = max_requests_per_minute
    available_token_capacity = max_tokens_per_minute
    last_update_time = time.time()

    # initialize flags
    file_not_finished = True  # after file is empty, we'll skip reading it
    logging.debug(f"Initialization complete.")

    # initialize file reading
    with open(requests_filepath) as file:
        # `requests` will provide requests one at a time
        requests = file.__iter__()
        logging.debug(f"File opened. Entering main loop")

        while True:
            # get next request (if one is not already waiting for capacity)
            if next_request is None:
                if not queue_of_requests_to_retry.empty():
                    next_request = queue_of_requests_to_retry.get_nowait()
                    logging.debug(f"Retrying request {next_request.task_id}: {next_request}")
                elif file_not_finished:
                    try:
                        # get new request
                        request_json = json.loads(next(requests))
                        tokens_to_be_consumed = num_tokens_consumed_from_request(request_json, api_endpoint, token_encoding_name)
                        next_request = APIRequest(
                            task_id=next(task_id_generator),
                            request_json=request_json,
                            token_consumption=tokens_to_be_consumed,
                            attempts_left=max_attempts,
                            print_interval = print_interval
                        )
                        status_tracker.num_tasks_started += 1
                        status_tracker.num_tasks_in_progress += 1
                        logging.debug(f"Reading request {next_request.task_id}: {next_request}")
                    except StopIteration:
                        # if file runs out, set flag to stop reading it
                        logging.debug("Read file exhausted")
                        file_not_finished = False

            # update available capacity
            current_time = time.time()
            seconds_since_update = current_time - last_update_time
            available_request_capacity = min(
                available_request_capacity + max_requests_per_minute * seconds_since_update / 60.0,
                max_requests_per_minute,
            )
            available_token_capacity = min(
                available_token_capacity + max_tokens_per_minute * seconds_since_update / 60.0,
                max_tokens_per_minute,
            )

            
            last_update_time = current_time

            # if enough capacity available, call API
            if next_request:
                next_request_tokens = next_request.token_consumption
                if (
                    available_request_capacity >= 1
                    and available_token_capacity >= next_request_tokens
                ):  
                    #print(int(available_request_capacity), int(available_token_capacity))
                    # update counters
                    available_request_capacity -= 1
                    available_token_capacity -= next_request_tokens
                    next_request.attempts_left -= 1

                    # call API
                    asyncio.create_task(
                        next_request.call_api(
                            request_url=request_url,
                            request_header=request_header,
                            retry_queue=queue_of_requests_to_retry,
                            save_filepath=save_filepath,
                            status_tracker=status_tracker,
                            print_interval=print_interval
                        )
                    )
                    next_request = None  # reset next_request to empty

            # if all tasks are finished, break
            if status_tracker.num_tasks_in_progress == 0:
                break

            # main loop sleeps briefly so concurrent tasks can run
            await asyncio.sleep(seconds_to_sleep_each_loop)

            # if a rate limit error was hit recently, pause to cool down
            seconds_since_rate_limit_error = (time.time() - status_tracker.time_of_last_rate_limit_error)
            if seconds_since_rate_limit_error < seconds_to_pause_after_rate_limit_error:
                remaining_seconds_to_pause = (seconds_to_pause_after_rate_limit_error - seconds_since_rate_limit_error)
                await asyncio.sleep(remaining_seconds_to_pause)
                # ^e.g., if pause is 15 seconds and final limit was hit 5 seconds ago
                logging.warn(f"Pausing to cool down until {time.ctime(status_tracker.time_of_last_rate_limit_error + seconds_to_pause_after_rate_limit_error)}")

        # after finishing, log final status
        logging.info(f"""Parallel processing complete. Results saved to {save_filepath}""")
        if status_tracker.num_tasks_failed > 0:
            logging.warning(f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed. Errors logged to {save_filepath}.")
        if status_tracker.num_rate_limit_errors > 0:
            logging.warning(f"{status_tracker.num_rate_limit_errors} rate limit errors received. Consider running at a lower rate.")


# dataclasses

@dataclass
class StatusTracker:
    """Stores metadata about the script's progress. Only one instance is created."""

    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0  # script ends when this reaches 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0  # excluding rate limit errors, counted above
    num_other_errors: int = 0
    time_of_last_rate_limit_error: int = 0  # used to cool off after hitting rate limits


@dataclass
class APIRequest:
    """Stores an API request's inputs, outputs, and other metadata. Contains a method to make an API call."""

    task_id: int
    request_json: dict
    token_consumption: int
    attempts_left: int
    result = []
    print_interval: int

    async def call_api(
        self,
        request_url: str,
        request_header: dict,
        retry_queue: asyncio.Queue,
        save_filepath: str,
        status_tracker: StatusTracker,
        print_interval: int,
    ):
        """Calls the OpenAI API and saves results."""
        if print_interval != 0:
            if self.task_id % print_interval == 0:
                logging.info(f"Starting request #{self.task_id}")
                
        error = None

        extra_id = self.request_json["extra_id"]
        request_json_clean = copy.deepcopy(self.request_json)
        del request_json_clean["extra_id"]

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url=request_url, headers=request_header, json=request_json_clean
                ) as response:
                    response = await response.json()
            if "error" in response:
                logging.warning(
                    f"Request {self.task_id} failed with error {response['error']}"
                )
                status_tracker.num_api_errors += 1
                error = response
                if "Rate limit" in response["error"].get("message", ""):
                    status_tracker.time_of_last_rate_limit_error = time.time()
                    status_tracker.num_rate_limit_errors += 1
                    status_tracker.num_api_errors -= 1  # rate limit errors are counted separately

        except Exception as e:  # catching naked exceptions is bad practice, but in this case we'll log & save them
            logging.warning(f"Request {self.task_id} failed with Exception {e}")
            status_tracker.num_other_errors += 1
            error = e
        if error:
            self.result.append(error)
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                logging.error(f"Request {self.request_json} failed after all attempts. Saving errors: {self.result}")
                append_to_jsonl([self.request_json, self.result, extra_id], save_filepath)
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
        else:
            append_to_jsonl([self.request_json, response, extra_id], save_filepath)
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1
            logging.debug(f"Request {self.task_id} saved to {save_filepath}")


# functions


def api_endpoint_from_url(request_url):
    """Extract the API endpoint from the request URL."""
    match = re.search('^https://[^/]+/v\\d+/(.+)$', request_url)
    return match[1]

def append_to_jsonl(data, filename: str) -> None:
    """Append a json payload to the end of a jsonl file."""
    json_string = json.dumps(data)
    with open(filename, "a") as f:
        f.write(json_string + "\n")


def num_tokens_consumed_from_request(
    request_json: dict,
    api_endpoint: str,
    token_encoding_name: str,
):
    """Count the number of tokens in the request. Only supports completion and embedding requests."""
    encoding = tiktoken.get_encoding(token_encoding_name)
    # if completions request, tokens = prompt + n * max_tokens
    if api_endpoint.endswith("completions"):
        max_tokens = request_json.get("max_tokens", 15)
        n = request_json.get("n", 1)
        completion_tokens = n * max_tokens

        # chat completions
        if api_endpoint.startswith("chat/"):
            num_tokens = 0
            for message in request_json["messages"]:
                num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":  # if there's a name, the role is omitted
                        num_tokens -= 1  # role is always required and always 1 token
            num_tokens += 2  # every reply is primed with <im_start>assistant
            return num_tokens + completion_tokens
        # normal completions
        else:
            prompt = request_json["prompt"]
            if isinstance(prompt, str):  # single prompt
                prompt_tokens = len(encoding.encode(prompt))
                num_tokens = prompt_tokens + completion_tokens
                return num_tokens
            elif isinstance(prompt, list):  # multiple prompts
                prompt_tokens = sum([len(encoding.encode(p)) for p in prompt])
                num_tokens = prompt_tokens + completion_tokens * len(prompt)
                return num_tokens
            else:
                raise TypeError('Expecting either string or list of strings for "prompt" field in completion request')
    # if embeddings request, tokens = input tokens
    elif api_endpoint == "embeddings":
        input = request_json["input"]
        if isinstance(input, str):  # single input
            num_tokens = len(encoding.encode(input))
            return num_tokens
        elif isinstance(input, list):  # multiple inputs
            num_tokens = sum([len(encoding.encode(i)) for i in input])
            return num_tokens
        else:
            raise TypeError('Expecting either string or list of strings for "inputs" field in embedding request')
    # more logic needed to support other API calls (e.g., edits, inserts, DALL-E)
    else:
        raise NotImplementedError(f'API endpoint "{api_endpoint}" not implemented in this script')


def task_id_generator_function():
    """Generate integers 0, 1, 2, and so on."""
    task_id = 0
    while True:
        yield task_id
        task_id += 1