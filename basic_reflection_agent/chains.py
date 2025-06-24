from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
load_dotenv()

class Chain : 
    def __init__(self) -> None:
        self.llm = ChatGoogleGenerativeAI(model = "gemini-2.0-flash")

        self.generation_prompt_template = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a Twitter tech influencer assistant. "
                        "Your job is to generate the best possible tweet for the user's request, "
                        "using all previous critiques as feedback. "
                        "Do NOT respond conversationally. "
                        "Only output the improved tweet, nothing else."
                    ),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )
        
        self.critique_prompt_template = ChatPromptTemplate.from_messages(
                        [
                            (
                                "system",
                                "You are a viral Twitter influencer and expert tweet critic. "
                                "Your job is to critique ONLY the most recent tweet in the conversation. "
                                "Provide detailed, actionable feedback for improvement. "
                                "Do NOT respond conversationally. "
                                "Do NOT generate a new tweet. "
                                "Only output the critique and recommendations."
                            ),
                            MessagesPlaceholder(variable_name="messages"),
                        ]
                    )

    
    def generation_chain(self) : 
        chain = (self.generation_prompt_template|self.llm)
        return chain
    def critique_chain(self) : 
        chain = (self.critique_prompt_template|self.llm)
        return chain
