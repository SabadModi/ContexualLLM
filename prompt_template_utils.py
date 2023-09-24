from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate


# for llama2
system_prompt = """ You are a helpful, respectful and honest assistant. 
    Always answer as helpfully as possible, while being safe. Your answers should not include any harmful
    , unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses 
    are socially unbiased and positive in nature. If a question does not make any sense, 
    or is not factually coherent, explain why instead of answering something not correct. 
    If you don't know the answer to a question, please don't share false information.
"""


def get_prompt_template(system_prompt=system_prompt, promptTemplate_type=None, history=False):

    if promptTemplate_type=="llama":
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        SYSTEM_PROMPT = B_SYS + system_prompt + E_SYS
        instruction = """
        Context: {history} \n {context}
        User: {question}"""

        prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
        prompt = PromptTemplate(input_variables=["history", "context", "question"], template=prompt_template)
        
    else:
        # for vicuna
        system_prompt = """A chat between a curious user and an artificial intelligence assistant. 
            The assistant gives helpful, detailed, and polite answers to the user's questions. 
        """
        prompt_template = system_prompt + """

        Context: {history} \n {context}
        User: {question}
        Answer:"""
        prompt = PromptTemplate(input_variables=["history", "context", "question"], template=prompt_template)
    
    memory = ConversationBufferMemory(input_key="question", memory_key="history")

    return prompt, memory, 
