# import openai
#
# # for backward compatibility, you can still use `https://api.deepseek.com/v1` as `base_url`.
# openai.api_base="https://api.deepseek.com"
# openai.api_key="sk-40c9be0bc5164e1fa12fdca928aa8004"
# # client = OpenAI(api_key="<your API key>", base_url="https://api.deepseek.com")
#
# response = openai.ChatCompletion.create(
#     model="deepseek-chat",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant"},
#         {"role": "user", "content": "帮我规划一下春节假期旅游"},
#   ],
#     max_tokens=1024,
#     temperature=0.7,
#     stream=False
# )
#
# print(response.choices[0].message.content)




# # 导入PyTorch库
# import torch
#
# # 打印可用的GPU设备数量
# print(torch.cuda.device_count())
#
# # 打印是否可以使用CUDA，即是否可以在GPU上运行计算
# print(torch.cuda.is_available())
#
# # 打印torch的版本
# print(torch.__version__)
#
# # 打印是否可以使用cuDNN，这是一个用于深度神经网络的库，它提供了优化的计算和内存访问模式
# print(torch.backends.cudnn.is_available)
#
# # 打印CUDA的版本号
# print(torch.cuda_version)






from transformers import AutoTokenizer, AutoModelForCausalLM
from models.deepseek import deepseekLoader
import torch
# # 加载模型和分词器
# # model_name = "deepseek-ai/deepseek-r1-distill-qwen-7b"
# print(torch.cuda.is_available())
# model_path=("D:\Documents\Scholars\codes\deepseek-aiDeepSeek-R1-Distill-Q"
#             "wen-1.5B")
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# messages = [
#         {"role": "system", "content": ""},
#         {"role": "user", "content": "You are an expert of log parsing, and now you will help to do log parsing."+"I want you to act like an expert of log parsing. I will give you a log message delimited by backticks. You must identify and abstract all the dynamic variables in logs with {placeholder} and output a static log template. Print the input log's template delimited by backticks."},
#         {"role": "assistant", "content": "Sure, I can help you with log parsing."},
#         ]
# examples = [{'query': 'Log message: `try to connected to host: 172.16.254.1, finished.`', 'answer': 'Log template: `try to connected to host: {ip_address}, finished.`'}]
# messages.append({"role": "user", "content": examples[0]['query']})
# messages.append(
#                 {"role": "assistant", "content": examples[0]['answer']})
# query="[10.30 16:49:06] chrome.exe - proxy.cse.cuhk.edu.hk:5070 close, 0 bytes sent, 0 bytes received, lifetime <1 sec"
# messages.append({"role": "user", "content": f"{query}\n<think>\n"})
# text = tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True
#     )
#
# model = AutoModelForCausalLM.from_pretrained(
#     model_path,
#     device_map="auto",
#     torch_dtype="auto"
# )
# inputs = tokenizer(text, return_tensors="pt").to(model.device)
# outputs = model.generate(
#         **inputs,
#         max_new_tokens=2048,
#         temperature=0.5
# )
# res=tokenizer.decode(outputs[0], skip_special_tokens=True)
# print(res)
#
# # 生成函数
# def generate(prompt):
#     inputs = tokenizer(text, return_tensors="pt").to(model.device)
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=2048,
#         temperature=0.6
#     )
#     print(outputs)
#     print(outputs.size())
#     print(outputs.shape)
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 测试
# print(generate("清明节旅游计划\n<think>\n"))








model_path="D:\Documents\Scholars\codes\deepseek-aiDeepSeek-R1-Distill-Qwen-1.5B"
modelLoader=deepseekLoader(model_path)

tokenizer=modelLoader.tokenizer
model=modelLoader.model
messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": "You are an expert of log parsing, and now you will help to do log parsing."+"I want you to act like an expert of log parsing. I will give you a log message delimited by backticks. You must identify and abstract all the dynamic variables in logs with {placeholder} and output a static log template. Print the input log's template delimited by backticks."},
        {"role": "assistant", "content": "Sure, I can help you with log parsing."},
        ]
examples = [{'query': 'Log message: `try to connected to host: 172.16.254.1, finished.`', 'answer': 'Log template: `try to connected to host: {ip_address}, finished.`'}]
messages.append({"role": "user", "content": examples[0]['query']})
messages.append(
                {"role": "assistant", "content": examples[0]['answer']})
query="[10.30 16:49:06] chrome.exe - proxy.cse.cuhk.edu.hk:5070 close, 0 bytes sent, 0 bytes received, lifetime <1 sec"
messages.append({"role": "user", "content": f"{query}\n<think>\n"})
text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

model_inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(
        **model_inputs,
        max_new_tokens=2048,
        temperature=0.5
)
res=tokenizer.decode(outputs[0], skip_special_tokens=True)
print(res)