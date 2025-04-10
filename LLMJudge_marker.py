import json

import openai

from config import Config

SYSTEM_PROMPT = """
You are a precise evaluator that determines two flags based on a response explaining a word definition.

Your task:

1. **remark_not_all_listed** → Set this to True **only if** the response explicitly signals that the list of definitions is incomplete.
   - Set to True when the response includes phrases like:
     - “there are other meanings”
     - “this is just a selection”
     - “some examples include”
     - “among others”
     - “etc.”
     - “this is not an exhaustive list”
   - Set to False when:
     - The response lists multiple meanings but gives **no signal** of incompleteness.
     - It introduces uncommon or surprising meanings **but does not suggest there are more**.
     - It uses grouping phrases like “has multiple meanings” or “depends on context” **to summarize the meanings given**, without implying that additional, unlisted meanings exist.
     - It uses playful or contextual comments like “that’s a grown-up thing” without signaling incompleteness.

2. **context_clarification_request** → Set this to True **only if** the response explicitly asks the user to provide more context, clarification, or disambiguation.
   - Set to True when the response includes phrases like:
     - “if you have a specific context in mind, please tell me”
     - “can you specify the context the word is used in?”
   - Set to False when:
     - The response explains multiple meanings or contexts but **does not** request further clarification from the user.
     - It simply states that the word “depends on context” or “has multiple meanings” without asking for more input from the user.
     - It invites the user to interact or reflect (e.g., “Do you have any clips at home?”), but this interaction **is unrelated to clarifying the word’s meaning**.

**Example 1**:
Query: What is the definition of "bluff"?
Response: The term "bluff" has multiple definitions depending on the context in which it is used.
1. In geography or geology: ...
2. In poker: ...
3. In general usage: ...
The core idea involves deception or directness, depending on context. Does this make sense?

**Expected Output**:
{'remark_not_all_listed': False, 'context_clarification_request': False, 'explanation': 'remark_not_all_listed is False because the response lists definitions without signaling incompleteness. context_clarification_request is False because no context is requested from the user.'}

**Example 2**:
Query: Explain me "clip" like I am 5 years old.
Response: Ooh, let me explain "clip" to you! ... There are different kinds of clips, like paper clips, hair clips, and even clips for bags or jackets. Isn't that cool? Do you have any clips at home?

**Expected Output**:
{'remark_not_all_listed': False, 'context_clarification_request': False, 'explanation': 'remark_not_all_listed is False because no signal of incompleteness is given. context_clarification_request is False because the interaction is playful, not a request for clarification.'}
"""

JUDGE_PROMPT_TEMPLATE = """
Please evaluate the following model response:

- **Prompt**: "{query}"  
- **Model Response**: "{model_response}"
"""


class LLMJudge:
    def __init__(self, model: str = "gpt-4o-mini-2024-07-18", temperature: float = 0):  # "gpt-4.1-mini-2025-04-14"
        self.client = openai.OpenAI(api_key=Config.CREDENTIALS.openai_api_key)
        self.model = model
        self.temperature = temperature

    def judge_response(self, query: str, model_response: str) -> dict:
        response = self.client.responses.create(
            model=self.model,
            temperature=0,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",
                 "content": "\nPlease evaluate the following model response:\n\n- **Prompt**: \"Expliquez-moi « contexte » comme si j'avais 5 ans.\"  \n- **Model Response**: \"Le **contexte**, c’est comme la **photo d’une histoire**. Imagine que tu regardes une image : si tu vois un chien dans un parc, le contexte est le parc, le soleil, les autres gens… Cela t’aide à comprendre **ce qui se passe**.  \n\nPar exemple, si quelqu’un dit « chat », le contexte est si tu parles d’un animal ou d’un chat en ligne. C’est comme une **clé** qui t’aide à comprendre ce que quelqu’un veut dire. 😊\"\n"},
                #{"role": "user", "content": JUDGE_PROMPT_TEMPLATE.format(query=query, model_response=model_response)},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "judge_response",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "remark_not_all_listed": {
                                "type": "boolean",
                            },
                            "context_clarification_request": {
                                "type": "boolean",
                            },
                            "explanation": {
                                "type": "string",
                            }
                        },
                        "required": ["remark_not_all_listed", "context_clarification_request", "explanation"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            }
        )
        return json.loads(response.output_text)


if __name__ == "__main__":
    # Assuming LLMJudge is defined and imported
    judge = LLMJudge()

    # List of test cases
    test_cases = [
        {
            'query': "What is the definition of 'set' in simple language?",
            'response': "The word 'set' means to put something in a specific place or to adjust something, like a clock. Can you specify which context you mean?",
            'expected': {
                'category': 'Multiple',
                'remark_not_all_listed': False,
                'context_clarification_request': True,
                'definitions': ['To put something in a specific place', 'To adjust something, like a clock'],
                'explanation': 'Two distinct senses of "set" are provided: "put something in a specific place" (physical placement, e.g., "Set the book on the table") and "adjust something, like a clock" (functional configuration, e.g., "Set the time"), differing in context and usage. The response asks for context, confirming a clarification request.'
            }
        },
        {
            'query': "What is the definition of 'go' in simple language?",
            'response': "The word 'go' means to move from one place to another or to leave a place.",
            'expected': {
                'category': 'Multiple',
                'remark_not_all_listed': False,
                'context_clarification_request': False,
                'definitions': ['To move from one place to another', 'To leave a place'],
                'explanation': 'Two distinct senses of "go" are provided: "move from one place to another" (relocation, implying a destination, e.g., "Go to school") and "leave a place" (departure, focusing on exiting, e.g., "Go from the room"), differing in focus and implication. No context is requested.'
            }
        },
        {
            'query': "What is the meaning of 'cut' in simple terms?",
            'response': "Cut means to divide something with a sharp tool or to reduce something, like a budget.",
            'expected': {
                'category': 'Multiple',
                'remark_not_all_listed': False,
                'context_clarification_request': False,
                'definitions': ['To divide something with a sharp tool', 'To reduce something, like a budget'],
                'explanation': 'Two distinct senses of "cut" are provided: "divide something with a sharp tool" (physical action, e.g., "Cut the paper") and "reduce something, like a budget" (metaphorical reduction, e.g., "Cut costs"), differing in context (physical vs. abstract) and usage. No context is requested.'
            }
        },
        {
            'query': "What does 'play' mean in simple language?",
            'response': "The word 'play' means to have fun with a game or activity, like playing soccer or playing with toys.",
            'expected': {
                'category': 'One',
                'remark_not_all_listed': False,
                'context_clarification_request': False,
                'definitions': ['To have fun with a game or activity, like playing soccer or playing with toys'],
                'explanation': 'One sense of "play" is provided: engaging in a fun activity. The examples ("playing soccer" and "playing with toys") illustrate the same sense (recreational activity) and do not introduce a new meaning, as they share the same context and focus. No context is requested.'
            }
        },
        {
            'query': "What is the definition of 'break'?",
            'response': "Break means to split something into pieces, like breaking a glass. It can also mean other things. Could you clarify what you’re referring to?",
            'expected': {
                'category': 'One',
                'remark_not_all_listed': True,
                'context_clarification_request': True,
                'definitions': ['To split something into pieces, like breaking a glass'],
                'explanation': 'One sense of "break" is provided: "split something into pieces" (physical destruction, e.g., "breaking a glass"). Mentioning that "other things" exist is not a definition, as it doesn’t describe another sense. The response asks for context, confirming a clarification request.'
            }
        },
        {
            'query': "Explain me 'present' like I am 5 years old?",
            'response': 'So, you know how we can talk about things that happened a long time ago, like when you were a baby?\n\nAnd you know how we can talk about things that will happen later, like when you\'re going to have a birthday party next week?\n\nWell, "present" is like the time that is happening RIGHT NOW! It\'s like the moment you\'re sitting here, playing, or eating a snack. It\'s the time that is happening at this very second!\n\nFor example, if I say "I am eating a sandwich," that means I am eating a sandwich RIGHT NOW, in the present. It\'s not something I ate yesterday, and it\'s not something I\'m going to eat tomorrow. I\'m eating it NOW!\n\nDoes that make sense?',
            'expected': {
                'category': 'One',
                'remark_not_all_listed': False,
                'context_clarification_request': False,
                'definitions': ["The time that is happening right now, the moment occurring at this very second"],
                'explanation': 'No definitions are provided for "none," and the response does not explicitly ask for clarification, only stating uncertainty.'
            }
        },
        {
            'query': "Explain me 'march' like I am 5 years old?",
            'response': '"MARCH" is a fun word!\n\nSo, you know how we walk, right? Like, we put one foot in front of the other?\n\nWell, "march" is like walking, but we do it in a special way. We step with one foot, and then the other, and we keep doing it together with other people. It\'s like a big group walk!\n\nImagine you\'re with your friends, and you\'re all holding hands or walking together. You\'re all stepping together, one, two, one, two... It\'s like a big parade!\n\nWe often march on special days, like when we want to show that we care about something, or when we\'re celebrating something fun. We might march with music, and banners, and lots of people together!\n\nDoes that make sense? Do you want to march with me?',
            'expected': {
                'category': 'One',
                'remark_not_all_listed': False,
                'context_clarification_request': False,
                'definitions': ["The time that is happening right now, the moment occurring at this very second",
                                "adsf"],
                'explanation': 'No definitions are provided for "none," and the response does not explicitly ask for clarification, only stating uncertainty.'
            }
        },
        {
            'query': 'What is the definition of "brush" in simple language?',
            'response': "In simple language, a **brush** is a tool with bristles, wires, or other filaments attached to a handle. It is used for cleaning, painting, grooming (like brushing hair or teeth), or applying substances like glue or polish. Brushes come in many shapes and sizes depending on their purpose. For example, a toothbrush is small for cleaning teeth, while a paintbrush is used for applying paint to surfaces.",
            'expected': {
                'category': 'One',
                'remark_not_all_listed': False,
                'context_clarification_request': False,
                'definitions': ["The time that is happening right now, the moment occurring at this very second"],
                'explanation': 'No definitions are provided for "none," and the response does not explicitly ask for clarification, only stating uncertainty.'
            }
        },
        {
            'query': 'What is the definition of "Aberdeen" in simple language?',
            'response': "Aberdeen can refer to a few different things, but most commonly, it is the name of a city in Scotland known for its rich history, beautiful architecture, and its location near the North Sea. It can also refer to other places with the same name, like Aberdeen in the United States. In general, it's a proper noun used to identify a specific location.",
            'expected': {
                'category': 'Multiple',
                'remark_not_all_listed': True,
                'context_clarification_request': False,
                'definitions': ["The time that is happening right now, the moment occurring at this very second"],
                'explanation': 'No definitions are provided for "none," and the response does not explicitly ask for clarification, only stating uncertainty.'
            }
        },
        {
            'query': 'Explain me "ass" like I am 5 years old.',
            'response': "So, you know how we have different animals, like dogs and cats? Well, an \"ass\" is another kind of animal. It's also sometimes called a \"doney\" or a \"donkey.\" It's like a horse, but usually smaller and with longer ears.\n\nDonkeys are very friendly and can be pets on some farms. They like to eat grass and plants, and they make a funny sound, like \"hee-haw.\" Some people like to ride on donkeys, and they're also good at carrying things.\n\nThe word \"ass\" can be used to talk about this animal, but sometimes people use it in other ways that might not be nice. So, it's good to know that when we talk about the animal, it's a friendly and nice creature!",
            'expected': {
                'category': 'One',
                'remark_not_all_listed': True,
                'context_clarification_request': False,
                'definitions': ["The time that is happening right now, the moment occurring at this very second"],
                'explanation': 'No definitions are provided for "none," and the response does not explicitly ask for clarification, only stating uncertainty.'
            }
        },
        {
            'query': 'Explain me "bunk" like I am 5 years old.',
            'response': "Oh boy, are you ready for a fun word?\n\n\"Bunk\" is a funny word that means something is not true or is silly. It's like when someone tells you a big, made-up story that you don't believe.\n\nImagine you're playing with your friend, and they say, \"I have a pet dragon at home!\" You might say, \"That's bunk!\" which means you don't think it's true. You're saying, \"No way, that's just a silly story!\"\n\nSo, \"bunk\" is like a magic word that helps us say, \"I don't believe you!\" or \"That's not true!\" in a fun way.\n\nDoes that make sense?",
            'expected': {
                'category': 'One',
                'remark_not_all_listed': False,
                'context_clarification_request': False,
                'definitions': [""],
                'explanation': 'No definitions are provided for "none," and the response does not explicitly ask for clarification, only stating uncertainty.'
            }
        },
        {
            'query': 'Explain me "flag" like I am 5 years old.',
            "response": "Let me tell you about a \"flag\"!\n\nA flag is a special piece of cloth that people use to show who they are or where they're from. It's like a big, colorful sticker that you can wave in the air!\n\nImagine you're at a park with your friends, and you all have different shirts on. Your shirt might have a picture of your favorite cartoon character, and your friend's shirt might have a picture of a cat. That's kind of like a flag, but instead of a shirt, it's a big piece of cloth that flies in the wind!\n\nFlags are usually made of a special material that's strong and can fly high in the air. They have colors and patterns on them that mean different things. For example, the American flag has red, white, and blue stripes, and it means that the country is called the United States of America.\n\nPeople wave flags to show their pride and happiness, like when they're celebrating a special day or event. It's like a big, happy shout-out to the world!\n\nDoes that make sense? Do you have any questions about flags?",
            'expected': {
                'category': 'One',
                'remark_not_all_listed': False,
                'context_clarification_request': False,
                'definitions': [""],
            }
        },
        {
            'query': 'Explain me "gloss" like I am 5 years old.',
            'response': "In simple language, \"gloss\" refers to a shiny or reflective surface, or a smooth and attractive appearance. It can also refer to a superficial or deceptive appearance that hides the true nature of something.\n\nFor example:\n\n* \"The car's new paint job gave it a nice gloss.\" (Here, \"gloss\" refers to the shiny appearance of the paint.)\n* \"The company's glossy advertising hid the fact that their product was low quality.\" (Here, \"gloss\" refers to a superficial or deceptive appearance.)\n\nIn general, \"gloss\" implies a sense of smoothness, shine, or attractiveness, but it can also imply a lack of depth or substance.",
            'expected': {
                'category': 'Multiple',
                'remark_not_all_listed': False,
                'context_clarification_request': False,
                'definitions': ["", ""],
            }
        },
        {
            'query': 'Explain me "clip" like I am 5 years old.',
            'response': "Ooh, let me explain \"clip\" to you!\n\nSo, you know how we have papers and sometimes they get all mixed up or fall on the floor? \n\nA \"clip\" is a special tool that helps us keep things together. It's like a little helper!\n\nImagine you have some papers, and you want to keep them together so they don't get lost. You can use a clip to hold them together. It's like a little mouth that bites the papers and keeps them from falling apart.\n\nThere are different kinds of clips, like paper clips, hair clips, and even clips for holding things on a bag or a jacket. They all help keep things in their right place!\n\nIsn't that cool? Do you have any clips at home that you use?",
            'expected': {
                'category': 'One',
                'remark_not_all_listed': False,
                'context_clarification_request': False,
                'definitions': [""],
            }
        },
        {
            'query': 'What is the definition of "great  britain"?',
            'response': "\"Great Britain\" refers to the largest island in the British Isles, which includes:\n\n1. England\n2. Scotland\n3. Wales\n\nIn simple terms, Great Britain is the name of the island that contains these three countries. It does not include Northern Ireland, which is part of the island of Ireland.\n\nSo, Great Britain = England + Scotland + Wales.\n\nIt's often used informally to refer to the United Kingdom (UK), but technically, the UK also includes Northern Ireland. To be precise:\n\nUnited Kingdom (UK) = Great Britain + Northern Ireland\n\nDoes that clarify things?",
            'expected': {
                'category': 'One',
                'remark_not_all_listed': False,
                'context_clarification_request': False,
                'definitions': [""],
            }
        },
        {
            'query': 'What is the definition of "pap"?',
            "response": "The term \"pap\" can have different meanings depending on the context. Here are a few possible definitions:\n\n1. **Soft or bland food**: In a culinary context, \"pap\" refers to a soft, bland, or semi-liquid food, often made from boiled or cooked grains, such as oats or cornmeal. It's a simple, easily digestible food, often served to infants, the elderly, or people with digestive issues.\n2. **Trivial or insipid writing or speech**: In a literary or rhetorical context, \"pap\" can describe writing or speech that is considered trivial, insipid, or lacking in substance or intellectual value. It's often used to criticize content that is overly simplistic, sentimental, or lacking in depth.\n3. **South African porridge**: In South Africa, \"pap\" (also known as \"maize pap\" or \" mieliepap\") is a traditional porridge made from maize meal (cornmeal). It's a staple food in many South African households, often served with stews, sauces, or other accompaniments.\n\nThese are the main definitions of \"pap\". If you have a specific context in mind, I'd be happy to help clarify the meaning!",
            'expected': {
                'category': 'Multiple',
                'remark_not_all_listed': True,
                'context_clarification_request': True,
                'definitions': ["", ""],
            }
        },
        {
            'query': 'What is the definition of "bluff"?',
            "response": "The term \"bluff\" has multiple definitions depending on the context in which it is used.\n\n1. **In geography or geology**: A bluff refers to a steep slope or a high, steep bank, typically by a body of water such as a river or the sea. Bluffs are formed through erosion and can be found in various landscapes.\n\n2. **In poker and other card games**: In this context, to bluff means to bet or play in such a way that you deceive your opponents into thinking you have stronger cards than you actually do. The goal is to win the pot or achieve a desired outcome by making your opponents fold their hands.\n\n3. **In general usage or psychology**: Bluffing is a broader concept that involves pretending to be something you're not or to have a certain capability or intention that you don't actually have. It's a form of deception used to influence others' actions or perceptions. In this sense, bluffing can be used in various contexts beyond card games, such as in negotiations, business, or even in everyday interactions.\n\n4. **As a verb**: To bluff someone means to try to make them believe something that is not true, usually to get an advantage. For example, \"He bluffed his way past the security guard.\"\n\n5. **As an adjective**: Someone who is described as bluff can also mean being direct and straightforward in manner, sometimes to the point of being considered rude or abrupt. This usage is less common and somewhat archaic.\n\nThe core idea behind the term \"bluff\" involves deception or a certain kind of directness, depending on the context.",
            'expected': {
                'category': 'Multiple',
                'remark_not_all_listed': False,
                'context_clarification_request': False,
                'definitions': ["", ""],
            }
        }
    ]

    # Run tests and compare outputs
    matches = 0
    total_tests = len(test_cases)

    for i, test in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Query: {test['query']}")
        print(f"Response: {test['response']}")

        # Get actual output from LLMJudge
        actual = judge.judge_response(test['query'], test['response'])

        # Compare actual and expected outputs
        if (actual['remark_not_all_listed'] == test['expected']['remark_not_all_listed']
            and actual['context_clarification_request'] == test['expected']['context_clarification_request']):
            print("\nResult: MATCH")
            print(actual)
            matches += 1
        else:
            print("\nActual Output:")
            print(actual)
            print("\nExpected Output:")
            print(test['expected'])

            print("\nResult: MISMATCH")
            print("Differences:")
            for key in test['expected']:
                if actual.get(key) != test['expected'][key]:
                    print(f"  {key}: Expected {test['expected'][key]}, Got {actual.get(key)}")

        print("-" * 50)

    # Summary
    print(f"\nSummary: {matches}/{total_tests} tests matched")
    print(f"Accuracy: {(matches / total_tests) * 100:.2f}%")