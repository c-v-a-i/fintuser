import json
from typing import List
import os
from collections import defaultdict, deque
from typing import Literal
import random
import string

from build_finetune_dataset import Message

MIN_REVIEW_LENGTH = 650


class MessageEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Message):
            return {
                "role": obj.role,
                "content": obj.content
            }
        return super().default(obj)



def keep_only_the_longest_assistant_message(messages: List[Message]) -> List[Message]:
    """
    Keeps the longest assistant message and combines any preceding user messages into one.
    """
    assistant_messages = [m for m in messages if m.role == 'assistant']
    if not assistant_messages:
        return []

    longest_message = max(assistant_messages, key=lambda m: len(m.content))
    longest_index = messages.index(longest_message)

    user_messages_before = [m.content for m in messages[:longest_index] if m.role == 'user']
    combined_user_message = Message(role='user', content=' '.join(user_messages_before)) if user_messages_before else None

    return [combined_user_message, longest_message] if combined_user_message else [longest_message]


def heuristically_filter_data(messages: List[Message]) -> List[Message]:
    """
    Heuristically filters the data to remove unwanted messages.
    Retains the list only if the total length of assistant messages is > MIN_REVIEW_LENGTH.
    """
    if sum(len(m.content) for m in messages if m.role == 'assistant') > MIN_REVIEW_LENGTH:
        return messages
    return []


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
    return the concatenation of all 'plain' text segments, adding 'blockquote' to them.
    """
    text_entities = message.get("text_entities", [])
    chunks = []
    for entity in text_entities:
        if entity.get("type") == "plain":
            chunks.append(entity.get("text", ""))
        if entity.get("type") == "blockquote":
            chunks.append(
                f'\n> {entity.get("text", "")}\n'
            )
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
    with open("../data/raw_chat_data/result.json", "r", encoding="utf-8") as f:
        data = json.load(f)

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
    cv_directory_path = "../data/files"
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
            child_texts: List[Message] = []
            for child_id in descendants:
                child_message = messages_dict[child_id]
                text_str = extract_plain_text(child_message)
                user_id = child_message['from_id']
                role = get_role(user_id, root_user_id)
                if text_str:
                    child_texts.append(Message(
                        role=role,
                        content=text_str
                    ))

            # Only if we have non-empty messages do we store in final_result
            if child_texts:
                final_result[str(root_id)] = {
                    "messages": child_texts,
                    "pdf_filepath": pdf_filepath
                }

    # ----------------------------------------------------------------------
    # TODO: go over all the records and filter out the instances
    #       based on the functions above
    # ----------------------------------------------------------------------
    # We'll iterate over each record in `final_result` and do two things:
    #   (1) Keep only the longest assistant message + combined user messages before it
    #   (2) Filter out any items that do not meet the heuristic length requirement.
    # If an item fails the filter (returns an empty list), we remove it from final_result.
    # ----------------------------------------------------------------------
    for root_id, item in list(final_result.items()):
        current_messages = item["messages"]
        # keep only the longest assistant message (plus preceding user messages)
        current_messages = keep_only_the_longest_assistant_message(current_messages)
        # apply heuristic filter (must exceed MIN_REVIEW_LENGTH in assistant messages)
        current_messages = heuristically_filter_data(current_messages)

        if not current_messages:
            del final_result[root_id]
        else:
            item["messages"] = current_messages

    # check if there are messages where assistant is not the last message
    for root_id, item in list(final_result.items()):
        current_messages = item["messages"]
        if current_messages[-1].role != 'assistant':
            print(f'Assistant is not the last message in the conversation', root_id, item["messages"])
            print('\n\n\n\n')
            # del final_result[root_id]

    # ----------------------------------------------------------------------
    # 6) OUTPUT / POST-PROCESS / SAVE RESULTS
    # ----------------------------------------------------------------------
    random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=5))
    result_filename = f"../data/json_files/pdf_children_texts_{random_string}_min{MIN_REVIEW_LENGTH}.json"
    with open(result_filename, "w", encoding="utf-8") as out_f:
        json.dump(final_result, out_f, ensure_ascii=False, indent=2, cls=MessageEncoder)

    print(f"Saved the result to {result_filename}")
    print(f"Total number of records: {len(final_result)}")
    print('Minimal assistant message length:', MIN_REVIEW_LENGTH)


if __name__ == "__main__":
    main()
