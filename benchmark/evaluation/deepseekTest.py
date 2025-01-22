import openai

# for backward compatibility, you can still use `https://api.deepseek.com/v1` as `base_url`.
openai.api_base="https://api.deepseek.com"
openai.api_key="sk-d4015dae044d45bea3fd24d701e2bd71"
# client = OpenAI(api_key="<your API key>", base_url="https://api.deepseek.com")

response = openai.ChatCompletion.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "帮我规划一下春节假期旅游"},
  ],
    max_tokens=1024,
    temperature=0.7,
    stream=False
)

print(response.choices[0].message.content)