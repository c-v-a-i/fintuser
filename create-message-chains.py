import json
import os
from collections import defaultdict, deque
from typing import Literal
import random
import string

def get_all_files_as_dictionary(directory_path: str) -> dict:
    """
    Return a dictionary {file_name: True} for all files in directory_path.
    This is a stand-in for your 'getAllFilesAsDictionary' logic.
    """
    return {
        f: True
        for f in os.listdir(directory_path)
        if os.path.isfile(os.path.join(directory_path, f))
    }

def is_message_with_cv(message: dict, cv_directory_content: dict) -> bool:
    """
    Checks whether the message is a 'CV' message.
    Conditions:
      - 'file_name' is in the cv_directory_content
      - 'mime_type' is exactly 'application/pdf'
    """
    file_name = message.get("file_name")
    mime_type = message.get("mime_type")
    if not file_name or not mime_type:
        return False
    return cv_directory_content.get(file_name, False) and (mime_type == "application/pdf")

def extract_plain_text(message: dict) -> str:
    """
    Given a message dict with a 'text_entities' field,
    return the concatenation of all 'plain' text segments.
    """
    text_entities = message.get("text_entities", [])
    chunks = []
    for entity in text_entities:
        if entity.get("type") == "plain":
            chunks.append(entity.get("text", ""))
    return "".join(chunks)


def get_role(user_id: str, root_user_id: str) -> Literal['assistant', 'user']:
    return 'user' if user_id == root_user_id else 'assistant'

def build_adjacency_list(messages):
    """
    Build a dictionary `graph` where graph[parent_id] = list of child_ids
    using the 'reply_to_message_id' field of each message.
    """
    graph = defaultdict(list)
    for msg in messages:
        parent_id = msg.get("reply_to_message_id")
        if parent_id is not None:
            graph[parent_id].append(msg["id"])
    return graph

def bfs_collect_subtree(root_id, graph):
    """
    Given a root_id and an adjacency list `graph`,
    return a list of *all* descendant message IDs in the subtree.
    """
    visited = set()
    queue = deque([root_id])
    descendants = []

    while queue:
        current_id = queue.popleft()
        for child_id in graph[current_id]:
            if child_id not in visited:
                visited.add(child_id)
                descendants.append(child_id)
                queue.append(child_id)

    return descendants

def main():
    # ----------------------------------------------------------------------
    # 1) LOAD THE TOP-LEVEL JSON
    # ----------------------------------------------------------------------
    with open("result.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    with open('pdf_children_texts.json', "r", encoding="utf-8") as f:
        ids_to_keep = { k: True for k in list(json.load(f).keys()) }
        print('ids to keep:',  ids_to_keep)

    # data now contains something like:
    # {
    #   "name": "Tech resume review",
    #   "type": "public_supergroup",
    #   "id": 1352932060,
    #   "messages": [ {...}, {...}, ... ]
    # }

    # Extract the messages array
    messages = data['messages'] 

    # ----------------------------------------------------------------------
    # 2) PREPARE A MESSAGES DICTIONARY FOR QUICK LOOKUPS
    # ----------------------------------------------------------------------
    messages_dict = {}
    for msg in messages:
        messages_dict[msg["id"]] = msg

    # ----------------------------------------------------------------------
    # 3) PREPARE CV DIRECTORY LOOKUP
    # ----------------------------------------------------------------------
    cv_directory_path = "files"
    cv_directory_content = get_all_files_as_dictionary(cv_directory_path)

    # ----------------------------------------------------------------------
    # 4) BUILD THE GRAPH: PARENT -> [CHILD_IDs]
    # ----------------------------------------------------------------------
    graph = build_adjacency_list(messages)

    # ----------------------------------------------------------------------
    # 5) COLLECT CHILDREN FOR EACH "CV" MESSAGE AND PREPARE FINAL STRUCTURE
    # ----------------------------------------------------------------------
    final_result = dict()  # Will map str(id) -> { messages: [...], pdf_filepath: ... }

    for msg in messages:
        if is_message_with_cv(msg, cv_directory_content):
            root_id = msg["id"]  # id of a message with CV pdf document
            root_user_id = msg.get('from_id')
            file_name = msg["file_name"]  # e.g. "my_resume.pdf"
            
            # Construct the PDF file path
            pdf_filepath = os.path.join(cv_directory_path, file_name)

            # Get all children in the subtree
            descendants = bfs_collect_subtree(root_id, graph)
            
            # Gather all plain text from those descendants
            child_texts = []
            for child_id in descendants:
                child_message = messages_dict[child_id]
                text_str = extract_plain_text(child_message)
                user_id = child_message['from_id']
                role = get_role(user_id, root_user_id)
                if text_str:
                    child_texts.append({
                        'role': role,
                        'content': text_str
                    })
                    # child_texts.append(text_str)

            # Only if we have non-empty messages do we store in final_result
            if child_texts:
                # Use string root_id for the JSON keys
                final_result[str(root_id)] = {
                    "messages": child_texts,
                    "pdf_filepath": pdf_filepath
                }



    final_result = {
        k: v for k, v in final_result.items() if k in ids_to_keep
    }

    # ----------------------------------------------------------------------
    # 6) OUTPUT / POST-PROCESS / SAVE RESULTS
    # ----------------------------------------------------------------------
    # final_result structure:
    #
    # {
    #   "123": {
    #     "messages": [
    #       "descendant message text #1", 
    #       "descendant message text #2", ...
    #     ],
    #     "pdf_filepath": "files/my_resume.pdf"
    #   },
    #   "456": { ... },
    #   ...
    # }
    #
    random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=5))

    result_filename = f"pdf_children_texts_{random_string}.json"
    with open(result_filename, "w", encoding="utf-8") as out_f:
        json.dump(final_result, out_f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
