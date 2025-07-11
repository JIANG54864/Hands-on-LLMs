from typing import NoReturn

from pydantic_ai.agent import AgentRunResult
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai import Agent
from pydantic_ai.providers.openai import OpenAIProvider
import tools

model = OpenAIModel(
    model_name='qwen3:8b',
    provider=OpenAIProvider(base_url='http://localhost:11434/v1')
)

# load_dotenv()
#
# model = OpenAIModel(
#     'deepseek-chat',
#     provider=OpenAIProvider(
#         base_url='https://api.deepseek.com', api_key=os.getenv("DEEPSEEK_API_KEY")
#     ),
# )

agent=Agent(model=model,
            system_prompt="用简洁的中文回答",
            tools= [tools.read_file, tools.list_files, tools.rename_file])
def main()->NoReturn:
    while True:
        user_input = input("Input: ")
        res:AgentRunResult[ str]=agent.run_sync(user_input)
        print(res.output)

if __name__ == '__main__':
    main()

