document = """
Dinosaurs were a dominant group of reptiles that lived during the Mesozoic Era, spanning approximately 230 million to 66 million years ago. Their existence is supported by extensive fossil evidence discovered on every continent, including Antarctica. One important fact is that the first scientifically described dinosaur, Megalosaurus, was named in 1824 by the British scientist Richard Owen, who later coined the term “Dinosauria,” meaning “terrible lizards.” Fossils such as bones, teeth, eggs, footprints, and even preserved skin impressions provide concrete proof of their diversity and biological complexity.

Dinosaurs varied greatly in size and diet. For example, Tyrannosaurus rex, one of the largest land predators, could grow up to about 12 meters long and had teeth measuring over 15 centimeters. In contrast, some dinosaurs like Compsognathus were roughly the size of a chicken. The largest known dinosaurs, such as Argentinosaurus, are estimated to have exceeded 30 meters in length and weighed more than 70 tons. Fossilized trackways show that certain species moved in herds, indicating social behavior, while nesting grounds discovered in Mongolia demonstrate that some dinosaurs laid eggs in organized colonies.

Scientific evidence also shows that birds evolved from small theropod dinosaurs. Fossils of feathered dinosaurs discovered in China, including specimens with preserved plumage, confirm this evolutionary link. The mass extinction event 66 million years ago, widely attributed to a massive asteroid impact near present-day Mexico, led to the disappearance of non-avian dinosaurs. However, birds remain as living descendants, meaning dinosaurs never fully vanished from Earth.
"""

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv
load_dotenv()

class Response(BaseModel):
    notes: str = Field(description="The revision notes created from the document.")
    quiz: str = Field(description="The quiz with 5 questions based on the document.")
    evaluation_score: str = Field(description="The score out of 10 evaluating the quality of the notes and quiz.")
    evaluation_explanation: str = Field(description="A brief explanation for the evaluation score.")


str_parser = StrOutputParser()
pydantic_output_parser = PydanticOutputParser(pydantic_object=Response)

closed_source_model = ChatOpenAI(model="gpt-4o-mini-2024-07-18")

llm = HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.1-8B-Instruct", task="text-generation")
open_source_model = ChatHuggingFace(llm=llm)

notes_prompt = PromptTemplate(template="Create the revision notes for the following document: {document}. The notes must be just limited and created from within the content being provided.", input_variables=["document"])
quiz_prompt = PromptTemplate(template="Create a quiz with 5 questions based on the following document : {document}. The questions must be just limited and created from within the content being provided.", input_variables=["document"])
evaluator_prompt = PromptTemplate(
    template="Evaluate the quality of the following notes and quiz: {notes}, {quiz} based on the following document: {document}. The evaluation must be just limited and created from within the content being provided. Provide a score out of 10 and a brief explanation for the score.{format_instructions}", 
    input_variables=["notes", "quiz", "document", "format_instructions"],
    partial_variables={"format_instructions": pydantic_output_parser.get_format_instructions()}
)

notes_quiz_chain = RunnableParallel(
    {
        'notes': notes_prompt | open_source_model | str_parser,
        'quiz': quiz_prompt | closed_source_model | str_parser,
        'document': RunnablePassthrough()
    }
)

evaluator_chain = evaluator_prompt | closed_source_model | pydantic_output_parser


full_chain = notes_quiz_chain | evaluator_chain

print(full_chain.get_graph().print_ascii())

response = full_chain.invoke({"document":document})
print(response)


