import unittest
from create_message_chains import keep_only_the_longest_assistant_message
from build_finetune_dataset import Message

class TestKeepOnlyTheLongestAssistantMessage(unittest.TestCase):
    def test_no_assistant_messages(self):
        messages = [
            Message(role='user', content='User message 1'),
            Message(role='user', content='User message 2')
        ]
        result = keep_only_the_longest_assistant_message(messages)
        self.assertEqual(result, [])

    def test_single_assistant_message(self):
        messages = [
            Message(role='assistant', content='Assistant message 1'),
            Message(role='user', content='User message 1'),
        ]
        result = keep_only_the_longest_assistant_message(messages)
        self.assertEqual(result, [messages[0]])

    def test_multiple_assistant_messages(self):
        messages = [
            Message(role='user', content='User message 1'),
            Message(role='assistant', content='Assistant message 1'),
        ]
        result = keep_only_the_longest_assistant_message(messages)
        self.assertEqual(result, [
            Message(role='user', content='User message 1'),
            Message(role='assistant', content='Assistant message 1'),
        ])

    def test_no_user_messages_before_longest_assistant(self):
        messages = [
            Message(role='assistant', content='Assistant message 1'),
            Message(role='assistant', content='Assistant message 2 is longer')
        ]
        result = keep_only_the_longest_assistant_message(messages)
        self.assertEqual(result, [messages[1]])

if __name__ == '__main__':
    unittest.main()