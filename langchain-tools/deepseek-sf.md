# Tool Calling and Structured Output

In this notebook, we will try to verify the tool calling and structured output capabilities of [DeepSeek V3](https://cloud.siliconflow.cn/open/models?target=deepseek-ai/DeepSeek-V3) and [DeepSeek R1](https://cloud.siliconflow.cn/open/models?target=deepseek-ai/DeepSeek-R1) models from SiliconFlow.

We will use both LangChain and LangGraph to test the tool calling and structured output capabilities of the two models.

```bash
# Get your API Key from: https://cloud.siliconflow.cn/account/ak
export DEEPSEEK_API_KEY=<your-siliconcloud-api-key>
export DEEPSEEK_API_BASE=https://api.siliconflow.cn/v1
```

## Defining Schemas


```python
from typing import Optional, Union
from pydantic import BaseModel, Field


# Tool Schemas
def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: First integer
        b: Second integer
    """
    return a + b


def multiply(a: int, b: int) -> int:
    """Multiply two integers.

    Args:
        a: First integer
        b: Second integer
    """
    return a * b

tools = [add, multiply]


# Structured Output
class Joke(BaseModel):
    """Joke to tell user."""

    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")
    rating: Optional[int] = Field(
        default=None, description="How funny the joke is, from 1 to 10"
    )


class ConversationalResponse(BaseModel):
    """Respond in a conversational manner. Be kind and helpful."""

    response: str = Field(description="A conversational response to the user's query")


class FinalResponse(BaseModel):
    final_output: Union[Joke, ConversationalResponse]


def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()
```


```python
## Test DeepSeek-V3
```


```python
query = "What is 3 * 12?"
```


```python
from langchain.chat_models import init_chat_model

# pip install -qU langchain-deepseek
llm = init_chat_model("deepseek-ai/DeepSeek-V3")
```


```python
# Tool calling test
llm_with_tools = llm.bind_tools(tools)
llm_with_tools.invoke(query)
```




    AIMessage(content='', additional_kwargs={'tool_calls': [{'id': '0195a3675a60caaef4f57654a565fd71', 'function': {'arguments': '{"a":3,"b":12}', 'name': 'multiply'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 22, 'prompt_tokens': 165, 'total_tokens': 187, 'completion_tokens_details': None, 'prompt_tokens_details': None}, 'model_name': 'deepseek-ai/DeepSeek-V3', 'system_fingerprint': '', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-928d89c0-825a-48c2-a798-be5e5d5d71da-0', tool_calls=[{'name': 'multiply', 'args': {'a': 3, 'b': 12}, 'id': '0195a3675a60caaef4f57654a565fd71', 'type': 'tool_call'}], usage_metadata={'input_tokens': 165, 'output_tokens': 22, 'total_tokens': 187, 'input_token_details': {}, 'output_token_details': {}})



✅ LangSmith Trace: https://smith.langchain.com/public/bb3e22bb-e580-4726-ba3d-3c21796282a1/r


```python
# Structured output test
structured_llm = llm.with_structured_output(FinalResponse)
structured_llm.invoke("Tell me a joke about cats")
```




    FinalResponse(final_output=Joke(setup="Why don't cats play poker in the jungle?", punchline='Too many cheetahs!', rating=None))



✅ LangSmith Trace: https://smith.langchain.com/public/b5a70e23-ab3f-4f04-9068-620b6efde5d0/r


```python
from langgraph.prebuilt import create_react_agent

# Tool calling test
graph = create_react_agent(llm, tools=tools)

inputs = {"messages": [("user", "what is 3 * 12?")]}
print_stream(graph.stream(inputs, stream_mode="values"))
```

    ================================[1m Human Message [0m=================================
    
    what is 3 * 12?
    ==================================[1m Ai Message [0m==================================
    Tool Calls:
      multiply (0195a38e69ff4c957317510af7423d63)
     Call ID: 0195a38e69ff4c957317510af7423d63
      Args:
        a: 3
        b: 12
    =================================[1m Tool Message [0m=================================
    Name: multiply
    
    36
    ==================================[1m Ai Message [0m==================================
    Tool Calls:
      multiply (0195a38e94f9b9ab02fa093a145563b9)
     Call ID: 0195a38e94f9b9ab02fa093a145563b9
      Args:
        a: 3
        b: 12
    =================================[1m Tool Message [0m=================================
    Name: multiply
    
    36
    ==================================[1m Ai Message [0m==================================
    
    Hello! How can I assist you today?


❓ LangSmith Trace: https://smith.langchain.com/public/190768d1-ed6e-4713-8613-5c6a1c0ae4c7/r


```python
graph = create_react_agent(
    llm,
    tools=tools,
    # specify the schema for the structured output using `response_format` parameter
    response_format=FinalResponse,
)

inputs = {"messages": [("user", "Tell me a joke about cats?")]}
graph.invoke(inputs)
```




    {'messages': [HumanMessage(content='Tell me a joke about cats?', additional_kwargs={}, response_metadata={}, id='995fdda3-c9b7-4c00-b1fc-950a3f9966b5'),
      AIMessage(content="Sure! Here's a joke about cats:\n\nWhy was the cat sitting on the computer?\n\nBecause it wanted to keep an eye on the mouse! 😹\n\nHope that gave you a chuckle! Let me know if you need anything else.", additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 48, 'prompt_tokens': 176, 'total_tokens': 224, 'completion_tokens_details': None, 'prompt_tokens_details': None}, 'model_name': 'deepseek-ai/DeepSeek-V3', 'system_fingerprint': '', 'finish_reason': 'stop', 'logprobs': None}, id='run-52c025f8-87ac-4385-b728-9ad07ac0d838-0', usage_metadata={'input_tokens': 176, 'output_tokens': 48, 'total_tokens': 224, 'input_token_details': {}, 'output_token_details': {}})],
     'structured_response': FinalResponse(final_output=Joke(setup='Why did the cat sit on the computer?', punchline='To keep an eye on the mouse!', rating=6))}



✅ LangSmith Trace: https://smith.langchain.com/public/29bbdccd-98e8-4aaf-a0db-4a092f361c9e/r

## Test DeepSeek-R1


```python
from langchain.chat_models import init_chat_model

# pip install -qU langchain-deepseek
llm = init_chat_model("deepseek-ai/DeepSeek-R1")
```


```python
# Tool calling test
llm_with_tools = llm.bind_tools(tools)
llm_with_tools.invoke(query)
```




    AIMessage(content='\n\nLet me calculate that for you.\n\n**Step 1:** I need to multiply 3 by 12.\n\n**Step 2:** Using the multiply function: multiply(a=3, b=12).\n\n**Result:** The result of 3 * 12 is 36.\n\n`multiply(3, 12) = 36`', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 71, 'prompt_tokens': 168, 'total_tokens': 239, 'completion_tokens_details': None, 'prompt_tokens_details': None}, 'model_name': 'deepseek-ai/DeepSeek-R1', 'system_fingerprint': '', 'finish_reason': 'stop', 'logprobs': None}, id='run-9cf121ca-14ff-4060-b0b9-7ca8eb9fc017-0', usage_metadata={'input_tokens': 168, 'output_tokens': 71, 'total_tokens': 239, 'input_token_details': {}, 'output_token_details': {}})



✅ LangSmith Trace: https://smith.langchain.com/public/0503ca60-1365-422e-9e78-5b85a92252a4/r


```python
# Structured output test
structured_llm = llm.with_structured_output(FinalResponse)
structured_llm.invoke("Tell me a joke about cats")
```




    FinalResponse(final_output=Joke(setup="Why don't cats play poker in the jungle?", punchline='Too many cheetahs!', rating=8))



✅ LangSmith Trace: https://smith.langchain.com/public/d849ba89-e77e-4c01-9bba-2ee874814121/r


```python
from langgraph.prebuilt import create_react_agent

# Tool calling test
graph = create_react_agent(llm, tools=tools)

inputs = {"messages": [("user", "what is 3 * 12?")]}
print_stream(graph.stream(inputs, stream_mode="values"))
```

    ================================[1m Human Message [0m=================================
    
    what is 3 * 12?
    ==================================[1m Ai Message [0m==================================
    
    
    
    Alright, let's solve 3 multiplied by 12. I need to calculate this using the `multiply` function. The parameters required are 'a' and 'b', both integers. Here, 3 and 12 are the integers to multiply.
    
    So, calling the multiply function with a=3 and b=12. Let me verify the calculation: 3 times 12 equals 36. Yes, that's correct. 
    
    **Step-by-Step Explanation:**
    - **Function Used:** `multiply(a: 3, b: 12)`
    - **Calculation:** 3 * 12 = 36
    - **Result:** 36
    
    **Answer:** The result of 3 multiplied by 12 is \boxed{36}.


❌ LangSmith Trace (Failed): https://smith.langchain.com/public/efa53517-d2be-434f-912b-eb708f47ffcc/r


```python
graph = create_react_agent(
    llm,
    tools=tools,
    # specify the schema for the structured output using `response_format` parameter
    response_format=FinalResponse,
)

inputs = {"messages": [("user", "Tell me a joke about cats?")]}
graph.invoke(inputs)
```




    {'messages': [HumanMessage(content='Tell me a joke about cats?', additional_kwargs={}, response_metadata={}, id='52bcebba-d74c-44f8-8272-0318258d2c6f'),
      AIMessage(content="\n\nSure! Here's a light-hearted cat joke for you:\n\nWhy don’t cats play poker in the jungle?  \n…Because there are too many *cheetahs*! 🐾😹", additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 43, 'prompt_tokens': 179, 'total_tokens': 222, 'completion_tokens_details': None, 'prompt_tokens_details': None}, 'model_name': 'deepseek-ai/DeepSeek-R1', 'system_fingerprint': '', 'finish_reason': 'stop', 'logprobs': None}, id='run-2e983ba0-f7ff-48d4-acd0-858b1c923db6-0', usage_metadata={'input_tokens': 179, 'output_tokens': 43, 'total_tokens': 222, 'input_token_details': {}, 'output_token_details': {}})],
     'structured_response': FinalResponse(final_output=Joke(setup="Why don't cats play poker in the jungle?", punchline='Too many cheetahs!', rating=None))}



✅ LangSmith Trace: https://smith.langchain.com/public/b13073d6-a96e-4be8-84fc-d361bd8089fd/r
