# LangChain for LLM Application Development

> Notes of Course by Deep Learning.AI
> 
> - [Course Link](https://learn.deeplearning.ai/langchain/lesson/1/introduction)
> - [Chinese Version by DataWhale](https://github.com/datawhalechina/prompt-engineering-for-developers/tree/main/content/LangChain%20for%20LLM%20Application%20Development)

- [LangChain for LLM Application Development](#langchain-for-llm-application-development)
  - [Model, Prompts and Parsers](#model-prompts-and-parsers)
    - [Prompt Template](#prompt-template)
    - [Output Parsers](#output-parsers)
  - [Memory](#memory)
    - [ConversationBufferMemory](#conversationbuffermemory)
    - [ConversationBufferWindowMemory](#conversationbufferwindowmemory)
    - [ConversationTokenBufferMemory](#conversationtokenbuffermemory)
    - [ConversationSummaryMemory](#conversationsummarymemory)
  - [Chains](#chains)
    - [LLMChain](#llmchain)
    - [Sequential Chains](#sequential-chains)
    - [SimpleSequentialChain](#simplesequentialchain)
    - [SequentialChain](#sequentialchain)
    - [Router Chain](#router-chain)
  - [Questions and Answer](#questions-and-answer)
  - [Evaluation](#evaluation)
  - [Agents](#agents)
  - [ReAct (Reason+Act) prompting in OpenAI GPT and LangChain](#react-reasonact-prompting-in-openai-gpt-and-langchain)


## Model, Prompts and Parsers

Set up

```py
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

#!pip install --upgrade langchain
from langchain.chat_models import ChatOpenAI
# To control the randomness and creativity of the generated
# text by an LLM, use temperature = 0.0
chat = ChatOpenAI(temperature=0.0)
chat
```

### [Prompt Template](https://python.langchain.com/en/latest/modules/prompts/chat_prompt_template.html)

```py
# template string
template_string = """Translate the text \
that is delimited by triple backticks \
into a style that is {style}. \
text: ```{text}```
"""

from langchain.prompts import ChatPromptTemplate
prompt_template = ChatPromptTemplate.from_template(template_string)

# check prompt 
prompt_template.messages[0].prompt

# check input variables
prompt_template.messages[0].prompt.input_variables

# create message from template

customer_style = """American English \
in a calm and respectful tone
"""

customer_email = """
Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls \
with smoothie! And to make matters worse, \
the warranty don't cover the cost of \
cleaning up me kitchen. I need yer help \
right now, matey!
"""

customer_messages = prompt_template.format_messages(
                    style=customer_style,
                    text=customer_email)

print(type(customer_messages)) #<class 'list'>
print(type(customer_messages[0])) #<class 'langchain.schema.HumanMessage'>


# Call the LLM to translate to the style of the customer message
customer_response = chat(customer_messages)

print(customer_response.content)
```

Why do we use prompt Tempaltes?

- When applied to more complex scenarios, prompts may be very long and involve many details. By using prompt templates, we can conveniently **reuse well-designed prompts**.
- LangChain also provides prompt templates for some common scenarios, such as **summarization, question answering, connecting to SQL databases, or connecting to different APIs**. By using the built-in prompt templates in LangChain, you can quickly build your own large-scale model applications without spending time designing and constructing prompts.
- When building large-scale model applications, we usually want the model's output to be in a given format, such as using specific keywords in the output to structure it. 
  - Below is an example of using a large model for chain thinking and reasoning. For the question "What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?"
  - by using LangChain library functions, the output uses "Thought", "Action", and "Observation" as keywords for chain thinking and reasoning, making the output structured. In the supplementary materials, you can see another code example of using LangChain and OpenAI for chain thinking and reasoning.

### [Output Parsers](https://python.langchain.com/en/latest/modules/prompts/output_parsers.html)

```py

# create a review template

customer_review = """\
This leaf blower is pretty amazing.  It has four settings:\
candle blower, gentle breeze, windy city, and tornado. \
It arrived in two days, just in time for my wife's \
anniversary present. \
I think my wife liked it so much she was speechless. \
So far I've been the only one using it, and I've been \
using it every other morning to clear the leaves on our lawn. \
It's slightly more expensive than the other leaf blowers \
out there, but I think it's worth it for the extra features.
"""

review_template_2 = """\
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product\
to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price,\
and output them as a comma separated Python list.

text: {text}

{format_instructions}
"""
prompt = ChatPromptTemplate.from_template(template=review_template_2)

# create output parser
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

gift_schema = ResponseSchema(name="gift",
                             description="Was the item purchased\
                             as a gift for someone else? \
                             Answer True if yes,\
                             False if not or unknown.")

delivery_days_schema = ResponseSchema(name="delivery_days",
                                      description="How many days\
                                      did it take for the product\
                                      to arrive? If this \
                                      information is not found,\
                                      output -1.")

price_value_schema = ResponseSchema(name="price_value",
                                    description="Extract any\
                                    sentences about the value or \
                                    price, and output them as a \
                                    comma separated Python list.")

response_schemas = [gift_schema, 
                    delivery_days_schema,
                    price_value_schema]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()
print(format_instructions)

# get message

messages = prompt.format_messages(text=customer_review, format_instructions=format_instructions)

# call chat

response = chat(messages)
print(response.content)

# use parser
type(output_dict)
output_dict = output_parser.parse(response.content)
output_dict

```

## [Memory](https://python.langchain.com/en/latest/modules/memory.html)

When interacting with language models, they do not remember previous conversations with you. This is a major issue when building applications such as chatbots, as it makes them less intelligent!

Therefore, in this section, we will introduce the Memory module in LangChain, and how it embeds previous conversations into the language model to make it capable of continuous dialogue.

When using the memory components in LangChain, they can help to save and manage historical chat messages, as well as build knowledge about specific entities. These components can store information across multiple rounds of dialogue and allow for tracking specific information and context during the conversation.

LangChain provides various types of memory, including:

- ConversationBufferMemory
- ConversationBufferWindowMemory
- Entity Memory
- Conversation Knowledge Graph Memory
- ConversationSummaryMemory
- ConversationSummaryBufferMemory
- ConversationTokenBufferMemory
- VectorStore-Backed Memory


Buffer memory allows for keeping track of recent chat messages, while summary memory provides a summary of the entire conversation. Entity memory allows for retaining information about specific entities across multiple rounds of dialogue.

These memory components are modular and can be combined with other components to enhance a chatbot's conversation management capabilities. The Memory module can be accessed and updated through simple API calls, allowing developers to more easily manage and maintain conversation histories.

This course mainly introduces four types of memory modules, and other modules can be learned from the documentation."

### ConversationBufferMemory

This type of memory allows for storing messages and retrieving them from variables.

```py
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

llm = ChatOpenAI(temperature=0.0,openai_api_key=OPENAI_API_KEY)  

# create buffer
memory = ConversationBufferMemory()
conversation = ConversationChain(   # Create a new conversation chain (we'll get into the details of chains later on)
    llm=llm, 
    memory = memory,
    verbose=True   # See what LangChain is actually doing (if set to False, only answers will be given and you won't see the green content below)
)

# Start conversation (three rounds)
conversation.predict(input="Hi, my name is Andrew")
conversation.predict(input="What is 1+1?")
conversation.predict(input="What is my name?")

# Extract historical messages
print(memory.buffer)   
# You can also print historical messages by using memory.load_memory_variables({})
memory.load_memory_variables({}) # The curly brackets here are actually an empty dictionary. There are more advanced features that allow users to use more complex inputs, but we won't discuss them in this short course, so don't worry about why there's an empty bracket here.

# Add specified input/output content to memory buffer
memory = ConversationBufferMemory()  # Create an empty conversation buffer memory
memory.save_context({"input": "Hi"},    # Add specified input/output to the buffer
                    {"output": "What's up"})

```

### ConversationBufferWindowMemory

As conversations become longer and longer, the amount of memory required also becomes very long. The cost of sending a large number of tokens to LLM also becomes more expensive, which is why API invocation fees are usually charged based on the number of tokens it needs to process.

To address these issues, LangChain also provides several convenient memories for storing historical conversations. Among them, conversation buffer window memory only keeps a buffer window of a certain size for conversation. It only uses the most recent n interactions. This can be used to maintain a sliding window of recent interactions so that the buffer does not become too large.

```py
from langchain.memory import ConversationBufferWindowMemory
memory = ConversationBufferWindowMemory(k=1)
memory.save_context({"input": "Hi"},
                    {"output": "What's up"})
memory.save_context({"input": "Not much, just hanging"},
                    {"output": "Cool"})
memory.load_memory_variables({})

```

### ConversationTokenBufferMemory

Using conversation token buffer memory, the memory will be limited to a certain number of saved tokens. If the number of tokens exceeds the specified limit, it will cut off the early parts of the conversation to retain the number of tokens corresponding to recent communication, but not exceeding the token limit.


```py
memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=30)
memory.save_context({"input": "AI is what?!"},
                    {"output": "Amazing!"})
memory.save_context({"input": "Backpropagation is what?"},
                    {"output": "Beautiful!"})
memory.save_context({"input": "Chatbots are what?"}, 
                    {"output": "Charming!"})

```

- ChatGPT uses a tokenization method based on Byte Pair Encoding (BPE) to split input text into tokens.
  - BPE is a common tokenization technique that splits input text into smaller subword units.
- OpenAI has released a new open-source Python library called "[tiktoken](https://github.com/openai/tiktoken)" on its official GitHub, which is mainly used for counting tokens. Compared to Hugging Face's tokenizer, it is several times faster. 
- For the specific token calculation methods, especially the difference between Chinese characters and English words, refer to [Link](https://www.zhihu.com/question/594159910)

### ConversationSummaryMemory

The idea behind this Memory is to not limit the memory to a fixed number of tokens or a fixed number of conversation turns based on the recent conversation, but instead to use LLM to generate summaries of the historical conversations up to the current point and store them.

```py
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain

# create a long string
schedule = "There is a meeting at 8am with your product team. \
You will need your powerpoint presentation prepared. \
9am-12pm have time to work on your LangChain \
project which will go quickly because Langchain is such a powerful tool. \
At Noon, lunch at the italian resturant with a customer who is driving \
from over an hour away to meet you to understand the latest in AI. \
Be sure to bring your laptop to show the latest LLM demo."

memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)   

memory.save_context({"input": "Hello"}, {"output": "What's up"})
memory.save_context({"input": "Not much, just hanging"},
                    {"output": "Cool"})
memory.save_context({"input": "What is on the schedule today?"}, 
                    {"output": f"{schedule}"})

memory.load_memory_variables({})

# create another conversation based on the memory
conversation = ConversationChain(  
    llm=llm, 
    memory = memory,
    verbose=True
)

conversation.predict(input="What would be a good demo to show?")
memory.load_memory_variables({})   #memory updated

```

## Chains

Why do we need Chains?

- Chains allow us to combine multiple components together to create a single, cohesive application. Typically, Chains combine an LLM (large language model) with prompts.
- Using this building block, you can also combine a bunch of these building blocks together to perform a series of operations on your text or other data. 
  - For example, we can create a chain that takes user input, formats it using prompt templates, and then passes the formatted response to an LLM. 
- We can build more complex Chains by combining multiple Chains together, or by combining Chains with other components.

### LLMChain

### Sequential Chains

### SimpleSequentialChain

### SequentialChain

### Router Chain


## Questions and Answer

## Evaluation

## Agents


## ReAct (Reason+Act) prompting in OpenAI GPT and LangChain

```py

#!pip install -q wikipedia

from langchain.docstore.wikipedia import Wikipedia
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, Tool, AgentExecutor
from langchain.agents.react.base import DocstoreExplorer

docstore=DocstoreExplorer(Wikipedia())
tools = [
  Tool(
    name="Search",
    func=docstore.search,
    description="Search for a term in the docstore.",
  ),
  Tool(
    name="Lookup",
    func=docstore.lookup,
    description="Lookup a term in the docstore.",
  )
]

# 
llm = OpenAI(
  model_name="gpt-3.5-turbo",
  temperature=0,
)

# initialize reAct Agent
react = initialize_agent(tools, llm, agent="react-docstore", verbose=True)
agent_executor = AgentExecutor.from_agent_and_tools(
  agent=react.agent,
  tools=tools,
  verbose=True,
)


question = "Author David Chanoff has collaborated with a U.S. Navy admiral who served as the ambassador to the United Kingdom under which President?"
agent_executor.run(question)

```
