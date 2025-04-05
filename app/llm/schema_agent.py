from app.model import model_hotel as models
from sqlalchemy.inspection import inspect
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate


def extract_schema_from_sqlalchemy():
    schema_text = ""
    for cls in models.Base.__subclasses__():
        table = cls.__table__
        schema_text += f"\nTable: {table.name}\n"
        for column in table.columns:    
            col_details = f"  - {column.name}: {column.type}"
            if column.primary_key:
                col_details += " (PK)"
            if column.foreign_keys:
                col_details += f" (FK: {list(column.foreign_keys)[0].target_fullname})"
            schema_text += col_details + "\n"
    return schema_text


def get_sqlalchemy_chain():
    schema = extract_schema_from_sqlalchemy()

    prompt_template = PromptTemplate(
        input_variables=["question"],
        template=f"""
            You are an intelligent assistant who answers user questions using the following database schema:

            {schema}

            Answer the following user question based on the schema:
            {{question}}
            """
    )

    llm = ChatOpenAI(model="gpt-4", temperature=0)

    def run_chain(question: str):
        prompt = prompt_template.format(question=question)
        response = llm.predict(prompt)
        return response

    return run_chain
