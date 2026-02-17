import os
from dotenv import load_dotenv
from typing import Optional
from pydantic import BaseModel
from langchain_openai import ChatOpenAI

load_dotenv()

class Assistant:

    def __init__(self, with_structured_output: Optional[bool] = False, response_format: Optional[BaseModel] = None):
        assert(with_structured_output != True or response_format is not None), "If with_structured_output is True, response_format must be provided"
        self.model_name = os.getenv("MODEL_NAME")
        self.base_url = os.getenv("BASE_URL")
        self.api_key = os.getenv("API_KEY")
        llm = ChatOpenAI(model_name=self.model_name, base_url=self.base_url, api_key=self.api_key)
        self.llm = llm.with_structured_output(response_format) if with_structured_output else llm

    async def run(self, messages):
        try:
            response = await self.llm.ainvoke(messages)
            return response

        except Exception as e:
            raise e
