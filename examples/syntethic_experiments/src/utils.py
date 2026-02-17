import json


def read_json(filename: str) -> dict:
    with open(filename, "r") as reader:
        return json.load(reader)


def write_json(filename: str, dictionary: dict) -> None:
    with open(filename, "w") as writer:
        json.dump(dictionary, writer, indent=4, sort_keys=True)
