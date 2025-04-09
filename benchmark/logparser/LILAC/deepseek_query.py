import re
import time
import string
import json
from .parsing_cache import ParsingCache
from .post_process import correct_single_template
from models.deepseek import deepseekLoader
def infer_llm(instruction, exemplars, query, log_message, tokenizer, model, temperature=0.6, max_tokens=2048):
    retry_times = 0
    print(f"Using model: deepseek")
    # 强制设置温度参数范围
    temperature = max(0.5, min(0.7, float(temperature)))
    print(f"Adjusted temperature to {temperature} per documentation")

    # messages = [
    #     {"role": "system", "content": ""},
    #     {"role": "user", "content": "You are an expert of log parsing, and now you will help to do log parsing."+instruction},
    #     {"role": "assistant", "content": "Sure, I can help you with log parsing."},
    #     ]
    messages=[
        {"role": "system", "content": ""},
        {"role": "user", "content": "你是一名计算机各类系统日志解析专家，拥有丰富的学识和日志解析经验，现在你将帮忙进行日志解析工作。我会给你一条由反引号界定的日志消息Log message ，你必须识别并将日志中所有的动态变量用 {placeholder} 进行抽象处理，然后输出一个静态的日志模板Log template。注意在给出最终结果时，请先输出“Log template: ”,随后打印出由反引号界定的输入日志的模板"},
        {"role": "assistant", "content": "好的，我会帮你进行日志解析。"},
    ]

    # print(exemplars)
    if exemplars is not None:
        for i, exemplar in enumerate(exemplars):
            messages.append({"role": "user", "content": exemplar['query']})
            messages.append(
                {"role": "assistant", "content": exemplar['answer']})
    # messages.append({"role": "user", "content": query})
    messages.append({"role": "user", "content": f"{query}\n<think>\n"})



    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # my_json = json.dumps(messages, ensure_ascii=False, separators=(',', ':'))
    # print(my_json)
    retry_times = 0
    while retry_times < 3:
        try:

            outputs = model.generate(
                **model_inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
            )

            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, outputs)
            ]

            # response1 = tokenizer.decode(outputs[0], skip_special_tokens=True)

            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]



            return response

        except Exception as e:
            print("Exception :", e)
            if "list index out of range" in str(e):
                break
            # print(answers)
            retry_times += 1

    print(f"Failed to get response from OpenAI after {retry_times} retries.")
    if exemplars is not None and len(exemplars) > 0:
        if exemplars[0]['query'] != 'Log message: `try to connected to host: 172.16.254.1, finished.`' \
                or exemplars[0]['answer'] != 'Log template: `try to connected to host: {ip_address}, finished.`':
            examples = [{'query': 'Log message: `try to connected to host: 172.16.254.1, finished.`',
                         'answer': 'Log template: `try to connected to host: {ip_address}, finished.`'}]
            return infer_llm(instruction, examples, query, log_message, tokenizer, model, temperature, max_tokens)
    return 'Log message: `{}`'.format(log_message)

def get_response_from_openai_key(query, tokenizer, model, examples=[], temperature=0.0):
    # Prompt-1
    # instruction = "I want you to act like an expert of log parsing. I will give you a log message enclosed in backticks. You must identify and abstract all the dynamic variables in logs with {placeholder} and output a static log template. Print the input log's template enclosed in backticks."
    instruction = "I want you to act like an expert of log parsing. I will give you a log message delimited by backticks. You must identify and abstract all the dynamic variables in logs with {placeholder} and output a static log template. Print the input log's template delimited by backticks."
    if examples is None or len(examples) == 0:
        examples = [{'query': 'Log message: `try to connected to host: 172.16.254.1, finished.`', 'answer': 'Log template: `try to connected to host: {ip_address}, finished.`'}]
    question = 'Log message: `{}`'.format(query)

    responses = infer_llm(instruction, examples, question, query, tokenizer, 
                          model, temperature, max_tokens=2048)
    return responses

def query_template_from_gpt(log_message, tokenizer, model, examples=[]):
    if len(log_message.split()) == 1:
        return log_message, False
    # print("prompt base: ", prompt_base)
    response = get_response_from_openai_key(log_message, tokenizer, model, examples)

    matches = re.split(r'\s*</think>\s*', response)
    final_answer = matches[-1].strip()


    # print(response)
    # lines = response.split('\n')
    lines = final_answer.split('\n')
    log_template = None
    for line in lines:
        if line.find("Log template:") != -1:
            log_template = line
            break
    if log_template is None:
        for line in lines:
            if line.find("`") != -1:
                log_template = line
                break
    if log_template is not None:
        start_index = log_template.find('`') + 1
        end_index = log_template.rfind('`')

        if start_index == 0 or end_index == -1:
            start_index = log_template.find('"') + 1
            end_index = log_template.rfind('"')

        if start_index != 0 and end_index != -1 and start_index < end_index:
            template = log_template[start_index:end_index]
            return template, True

    print("======================================")
    print("ChatGPT response format error: ")
    print(final_answer)
    print("======================================")
    return log_message, False


def post_process_template(template, regs_common):
    pattern = r'\{(\w+)\}'
    template = re.sub(pattern, "<*>", template)
    for reg in regs_common:
        template = reg.sub("<*>", template)
    template = correct_single_template(template)
    static_part = template.replace("<*>", "")
    punc = string.punctuation
    for s in static_part:
        if s != ' ' and s not in punc:
            return template, True
    print("Get a too general template. Error.")
    return "", False


def query_template_from_gpt_with_check(log_message, tokenizer, model, regs_common=[], examples=[]):
    template, flag = query_template_from_gpt(log_message, tokenizer, model, examples)
    if len(template) == 0 or flag == False:
        print(f"ChatGPT error")
    else:
        tree = ParsingCache()
        template, flag = post_process_template(template, regs_common)
        if flag:
            tree.add_templates(template)
            if tree.match_event(log_message)[0] == "NoMatch":
                print("==========================================================")
                print(log_message)
                print("ChatGPT template wrong: cannot match itself! And the wrong template is : ")
                print(template)
                print("==========================================================")
            else:
                return template, True
    return post_process_template(log_message, regs_common)