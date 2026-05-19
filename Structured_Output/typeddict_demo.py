from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int    


new_person: Person = {"name": "Yash", "age": 27}
print(new_person)