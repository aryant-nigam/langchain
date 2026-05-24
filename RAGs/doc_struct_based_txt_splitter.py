from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

text = """
class Student:
    def __init__(self, name, roll_no, marks):
        self.name = name
        self.roll_no = roll_no
        self.marks = marks

    def display_info(self):
        print("Name:", self.name)
        print("Roll No:", self.roll_no)
        print("Marks:", self.marks)

    def is_pass(self):
        if self.marks >= 40:
            return "Pass"
        else:
            return "Fail"


# Example usage
student1 = Student("Rahul", 101, 78)
student2 = Student("Anita", 102, 35)

student1.display_info()
print("Result:", student1.is_pass())

print()

student2.display_info()
print("Result:", student2.is_pass())
"""
chunks = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, chunk_size=450, chunk_overlap=0).split_text(text)
for c in chunks:
    print(c, end="\n\n______________________________\n\n")