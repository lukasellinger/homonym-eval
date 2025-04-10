import json

import openai

from config import Config

SYSTEM_PROMPT = """
You will extract definitions from the following response and determine whether it expresses one or multiple distinct meanings.

Instructions:

1. Read the response carefully.
2. Identify the distinct meanings or senses of the word being defined.

   - Treat it as **one definition** if the response elaborates, gives examples, describes subtypes, features, consequences, or mentions alternative forms **within the same core meaning**.

   - Split it into **multiple definitions** if the response **explicitly or implicitly includes clearly distinct meanings or senses**, such as:
     - Different referents (e.g., object vs. concept)
     - Informal, colloquial, or figurative uses that shift the meaning
     - Literal vs. metaphorical senses
     - Different grammatical or functional roles (e.g., noun vs. verb)
     - **Two usages with different goals or implications** (e.g., "move to a place" vs. "leave a place")

   - Pay attention to **compound sentences** or **"or"-phrases** that may compress multiple meanings into one line.

   - Do **not** split definitions if:
     - There are multiple examples or elaborations of the same concept
     - A second meaning is only vaguely referenced or not described
     - The distinctions are purely stylistic or emotional without changing the core use

3. Provide three outputs:
   - `definitions`: a list of distinct definitions (if any).
   - `category`: either 'One' or 'Multiple'.
   - `explanation`: a brief reason for your categorization.

Make sure your reasoning accounts for **semantic function**—not just phrasing. If two parts of the response describe actions or uses that occur in different contexts or with different implications, split them.

Example 1 (single meaning with details):
Input:
“March means to walk in a formal, organized way as part of a group, often with music and banners.”

Output:
definitions:
- To walk in a formal, organized way as part of a group, often with music and banners.
category: One
explanation: This is a single meaning with elaborative details; no distinct senses are presented.

Example 2 (clear multiple meanings):
Input:
“Break can mean to separate into pieces by force, or to pause from an activity.”

Output:
definitions:
- To separate into pieces by force.
- To pause from an activity.
category: Multiple
explanation: These are two clearly distinct senses of the word.

Example 3 (no split for subtype and mention):
Input:
“An ass is another kind of animal, also called a donkey. Sometimes people use the word in not-so-nice ways.”

Output:
definitions:
- An animal, also called a donkey, similar to a horse but smaller with longer ears.
category: One
explanation: Only one meaning is fully defined; the other usage is mentioned but not explained as a separate sense.

Example 4 (no split for examples and types):
Input:
“A clip is a special tool that helps keep things together, like papers, hair, or bags.”

Output:
definitions:
- A special tool that helps keep things together, like papers, hair, or bags.
category: One
explanation: These are examples and subtypes of the same core meaning.
"""

CHILD_SYSTEM_PROMPT = """
You are a precise evaluator tasked with analyzing a model response defining a word in a child-friendly manner (e.g., prompted with 'Explain me [word] like I am 5 years old'). Your goal is to extract distinct definitions and categorize the response as having one or multiple meanings, recognizing that child-friendly responses often use imaginative or example-heavy language to describe a single core meaning.

Instructions:

1. **Read the Response Carefully**:
   - Identify the word being defined and all meanings or senses provided. Note that child-friendly responses often include metaphors, playful language, or multiple examples to engage young audiences.

2. **Extract Definitions**:
   - **One Definition**: Count as one if the response describes a single core meaning, even if it includes:
     - Imaginative metaphors (e.g., 'a big hug from the water' for a bay).
     - Multiple examples or subtypes (e.g., 'paper clips, hair clips' for a clip).
     - Playful or emotional elaborations (e.g., 'makes you feel like you can fly' for a cape).
     - Descriptions of uses, contexts, or feelings tied to the same concept (e.g., 'worn by superheroes or at Halloween' for a cape).
   - **Multiple Definitions**: Split into separate definitions only if the response **explicitly introduces distinct meanings or senses**, such as:
     - Different referents (e.g., a physical object vs. an abstract concept).
     - Literal vs. metaphorical senses with clear shifts in meaning.
     - Different grammatical roles (e.g., noun vs. verb) with distinct implications.
   - **Do Not Split**:
     - Examples, subtypes, or contexts that support the same core concept (e.g., 'playing soccer or with toys' for play).
     - Stylistic, emotional, or imaginative variations that do not change the core sense (e.g., 'a magic cloak' for a cape).
   - **Compound Sentences or 'or'-Phrases**: Evaluate carefully. Only split if they describe distinct senses with different implications (e.g., 'to put or adjust' may be two senses if contexts differ significantly).
   - **Kid-Friendly Language**: Treat metaphors, analogies, or engaging descriptions as elaborations of a single meaning unless they clearly introduce a new sense. For example, 'a big hug from the water' and 'calm place for boats' describe the same geographic feature (bay).
   - Do not verify factual correctness—extract only what is present.

3. **Assign Category**:
   - **None**: No definitions provided.
   - **One**: One distinct definition extracted.
   - **Multiple**: Two or more distinct definitions extracted.

4. **Provide Outputs**:
   - `definitions`: A list of distinct definitions, summarizing each sense concisely.
   - `category`: 'None', 'One', or 'Multiple'.
   - `explanation`: A brief reason for the categorization, addressing why details are treated as elaborations or distinct senses in the child-friendly context.

Example 1 (Single Meaning with Kid-Friendly Elaborations):
Input: 'A bay is a special area where the ocean curves into the land. It’s like a big hug from the water! Bays are good places for boats because the water is calmer and protected from big ocean waves.'
Output:
{
  'definitions': ['A special area where the ocean curves into the land, providing a calm place protected from big waves'],
  'category': 'One',
  'explanation': 'The response describes one meaning of bay (a geographic feature) with kid-friendly metaphors (‘big hug’) and elaborations (calm water, boats). No distinct senses are introduced.'
}

Example 2 (Single Meaning with Playful Language):
Input: 'A cape is a special kind of clothing that you wear on your back! It’s like a big piece of fabric that goes around your neck and hangs down behind you. Imagine you’re a superhero, like Superman or Batman! You wear a cape to make you feel strong and brave. It’s like a magic cloak that helps you fly (well, not really, but it makes you feel like you can!). Some people wear capes on Halloween or at costume parties to dress up like their favorite superheroes.'
Output:
{
  'definitions': ['A special kind of clothing that you wear on your back, like a big piece of fabric that goes around your neck and hangs down behind you'],
  'category': 'One',
  'explanation': 'The response defines cape as clothing with kid-friendly elaborations (superhero imagery, Halloween use, emotional effects). These are not distinct senses but support the same concept.'
}

Example 3 (Single Meaning with Examples):
Input: 'Ooh, let me explain "clip" to you! A clip is a special tool that helps us keep things together. It’s like a little helper! You can use a clip to hold papers or keep hair in place. There are different kinds of clips, like paper clips or hair clips.'
Output:
{
  'definitions': ['A special tool that helps keep things together, like for papers or hair'],
  'category': 'One',
  'explanation': 'The response defines clip as a fastening tool with kid-friendly examples (papers, hair). These are elaborations of one meaning, not distinct senses.'
}

Example 4 (Multiple Meanings, Hypothetical):
Input: 'A bank is a place where you keep your money, like where your piggy bank savings go! It’s also a piece of land next to a river, like where you might have a picnic by the water.'
Output:
{
  'definitions': ['A place where money is kept, like a financial institution', 'A piece of land next to a river'],
  'category': 'Multiple',
  'explanation': 'The response provides two distinct meanings of bank (financial institution and river land), each with kid-friendly descriptions. These are separate senses with different referents.'
}

Example 5 (Single Meaning with Cultural Context):
Input: 'Oh boy, are you going to love learning about the Cheyenne! So, a long, long time ago, there was a group of Native American people called the Cheyenne. They lived in a place called the Great Plains, which is a big area of grasslands in the middle of the United States. The Cheyenne people were very good at riding horses and hunting animals like buffalo.'
Output:
{
  'definitions': ['A Native American tribe that lived in the Great Plains'],
  'category': 'One',
  'explanation': 'The response defines Cheyenne as a Native American tribe with kid-friendly details (Great Plains, horses, hunting). These are elaborations of one meaning, not distinct senses.'
}
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
                {"role": "system", "content": CHILD_SYSTEM_PROMPT},
                {"role": "user", "content": JUDGE_PROMPT_TEMPLATE.format(query=query, model_response=model_response)},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "judge_response",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "definitions": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                }
                            },
                            "category": {
                                "type": "string",
                                "enum": ["None", "One", "Multiple"]
                            },
                            "explanation": {
                                "type": "string",
                            }
                        },
                        "required": ["definitions", "category", "explanation"],
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
            'query': 'What is the definition of "great  britain"?',
            'response': "\"Great Britain\" refers to the largest island in the British Isles, which includes:\n\n1. England\n2. Scotland\n3. Wales\n\nIn simple terms, Great Britain is the name of the island that contains these three countries. It does not include Northern Ireland, which is part of the island of Ireland.\n\nSo, Great Britain = England + Scotland + Wales.\n\nIt's often used informally to refer to the United Kingdom (UK), but technically, the UK also includes Northern Ireland. To be precise:\n\nUnited Kingdom (UK) = Great Britain + Northern Ireland\n\nDoes that clarify things?",
            'expected': {
                'category': 'Multiple',
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
        },
    ]

    child_test_cases = [
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
            'query': 'Explain me "bay" like I am 5 years old.',
            "response": "So, you know how we have big oceans with lots of water, right? And sometimes the water goes into a special area that's kind of surrounded by land?\n\nThat special area is called a \"bay\"! It's like a big, curvy arm of the ocean that hugs the land. Imagine you're playing with a big bowl of water, and you make a little curve in the side of the bowl. That's kind of like a bay!\n\nBays are usually calm and peaceful, and lots of boats and ships like to sail in them because they're protected from the big waves in the ocean. Some bays are even connected to rivers, so you can see freshwater and saltwater mixing together!\n\nDoes that make sense?",
            'expected': {
                'category': 'One',
                'remark_not_all_listed': False,
                'context_clarification_request': False,
                'definitions': [""],
            }
        },
        {
            'query': 'Explain me "cape" like I am 5 years old.',
            "response": "A cape is a special kind of clothing that you wear on your back! It's like a big piece of fabric that goes around your neck and hangs down behind you.\n\nImagine you're a superhero, like Superman or Batman! You wear a cape to make you feel strong and brave. It's like a magic cloak that helps you fly (well, not really, but it makes you feel like you can!).\n\nSome people wear capes on Halloween or at costume parties to dress up like their favorite superheroes. And some people even wear capes just for fun, because they like the way it makes them feel!\n\nDoes that make sense? Do you want to be a superhero with a cape?",
            'expected': {
                'category': 'One',
                'remark_not_all_listed': False,
                'context_clarification_request': False,
                'definitions': [""],
            }
        },
        {
            'query': 'Explain me "canon" like I am 5 years old.',
            "response": "Oh boy, are you going to love this!\n\nSo, you know how there are lots of stories about your favorite superheroes or characters, like Batman or Elsa? And sometimes, you watch a movie or read a book about them, and it's really cool?\n\nWell, \"canon\" is like a special rule that helps us know which stories are the \"real\" ones, and which ones are just made up for fun.\n\nThink of it like a big box where we keep all the \"official\" stories about a character. If a story is \"in canon\", it means it's like it's been put in the box, and it's one of the \"true\" stories about that character.\n\nFor example, let's say you love Star Wars, and you watch a movie where Luke Skywalker fights Darth Vader. That's a \"canon\" story, because it's one of the official Star Wars movies.\n\nBut, if someone makes up a new story about Luke Skywalker, like \"Luke Skywalker has a pet dragon\", that's not \"in canon\", because it's not one of the official stories. It's like someone is just making it up for fun!\n\nSo, \"canon\" helps us know what's \"official\" and what's just made up. It makes it easier for us to know what's going on in the story, and to enjoy it more!\n\nDoes that make sense?",
            'expected': {
                'category': 'One',
                'remark_not_all_listed': False,
                'context_clarification_request': False,
                'definitions': [""],
            }
        },
        {
            'query': 'Explain me "cast" like I am 5 years old.',
            "response": "Oh boy, are you going to love this!\n\nImagine you have a toy box full of different toys like blocks, dolls, and cars. Now, let's say you want to play with the blocks, but they're all mixed up with the dolls and cars.\n\n\"Cast\" is like picking the toys you want to play with and putting them into a special group. It's like making a team of just the blocks, or just the cars.\n\nFor example, if you're playing with toy cars, you might say, \"I'm going to cast my favorite cars into a racing team!\" That means you're choosing the cars you want to be on your team.\n\nIn grown-up language, \"cast\" can mean a few different things, but it's often used when we're talking about choosing people or things to be part of a group, like a team or a movie. For example, when making a movie, the director might \"cast\" actors to play certain roles.\n\nSo, that's \"cast\" in a nutshell! It's like picking your favorite toys and making a special team with them.",
            'expected': {
                'category': 'One',
                'remark_not_all_listed': True,
                'context_clarification_request': False,
                'definitions': [""],
            }
        },
        {
            'query': 'Explain me "Cheyenne" like I am 5 years old.',
            "response": "Oh boy, are you going to love learning about the Cheyenne!\n\nSo, a long, long time ago, there was a group of Native American people called the Cheyenne. They lived in a place called the Great Plains, which is a big area of grasslands in the middle of the United States.\n\nThe Cheyenne people were very good at riding horses and hunting animals like buffalo. They would travel around in big groups, called tribes, and set up camps wherever they went. They would put up special tents called teepees, which were made from animal hides and sticks.\n\nThe Cheyenne were very good at making things too! They would make beautiful clothes, like buckskin dresses and feathered headdresses, and they would decorate their horses with colorful blankets and feathers.\n\nThe Cheyenne people loved to tell stories and sing songs together. They would have big gatherings, called powwows, where they would dance and sing and share food.\n\nThe Cheyenne were also very brave and strong. They would protect their families and homes from harm, and they were very good at fighting to keep their land safe.\n\nToday, there are still Cheyenne people living in the United States, and they continue to celebrate their traditions and culture. They are very proud of who they are and where they come from!\n\nIsn't that cool? Do you have any questions about the Cheyenne?",
            'expected': {
                'category': 'One',
                'remark_not_all_listed': False,
                'context_clarification_request': False,
                'definitions': [""],
            }
        }
    ]

    child_test_cases = [
        {
            'query': 'Explain me "china" like I am 5 years old.',
            "response": "Let me tell you about CHINA!\n\nChina is a very, very big country! It's so big that lots of people live there. Imagine a big playground with lots of swings, slides, and friends playing together. That's kind of like China!\n\nChina is on the other side of the world from where you live. If you were to travel there, you would have to go on a really, really long plane ride!\n\nIn China, people speak a different language called Mandarin. It's like a secret code that they use to talk to each other! \n\nChina is also famous for some yummy foods like noodles, dumplings, and rice. You might like to try some of these foods one day!\n\nAnd guess what? China is also home to some really cool animals like pandas! They're big, fluffy bears that love to eat bamboo.\n\nSo, that's China in a nutshell! It's a big, exciting country with lots of people, yummy food, and cool animals. Who knows, maybe one day you'll get to visit China and see all the amazing things it has to offer!",
            'expected': {
                'category': 'One',
                'remark_not_all_listed': False,
                'context_clarification_request': False,
                'definitions': [""],
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
        if (actual['category'] == test['expected']['category']
                and ((actual['category'] == 'One' and len(actual['definitions']) == 1)
                     or (actual['category'] == 'Multiple' and len(actual['definitions']) > 1)
                     or (actual['category'] == 'None' and len(actual['definitions']) == 0))):
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