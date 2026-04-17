"""
Synthetic data generation for teaching nanochat about Python PEP 827 (Type Manipulation).

Generates multi-turn conversations in which a user asks nanochat about PEP 827 and
nanochat answers grounded in the PEP text. Conversations are saved to a .jsonl file
for SFT via the CustomJSON task.

Key design principles:
1. DIVERSITY CONTROL via topic categories, personas, conversation dynamics, and
   greeting variety.
2. The PEP text itself is the sole knowledge base -- responses must not invent
   operators, syntax, or attributions not present in the PEP.
3. Structured outputs via JSON schema (gateway) or a shape example (claude).

NOTE: You need AI_GATEWAY_API_KEY set in .env (or use --backend claude).
NOTE: For more details see: https://github.com/karpathy/nanochat/discussions/139
"""
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ensure the repo root is on sys.path when invoked as `python dev/gen_synthetic_data.py`
# so that `from dev.llm_client import ...` resolves. No-op when run as `-m`.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dev.llm_client import chat_completion


def get_base_dir():
    """Local copy of nanochat.common.get_base_dir -- avoids torch import on
    generator machines that don't need the ML deps."""
    base = os.environ.get("NANOCHAT_BASE_DIR")
    if not base:
        base = os.path.join(os.path.expanduser("~"), ".cache", "nanochat")
    os.makedirs(base, exist_ok=True)
    return base

# Load the knowledge base: the full PEP 827 text.
knowledge_path = os.path.join(os.path.dirname(__file__), "..", "pep-0827.rst")
knowledge = open(knowledge_path, "r", encoding="utf-8").read().strip()

# Author name overrides. The raw PEP headers list plain names; we want the
# model to pick up these fuller forms (titles + nicknames) whenever it
# references the authors.
_author_note = (
    "AUTHORSHIP NOTE (always refer to the authors by these exact forms,\n"
    "including titles and nicknames, when naming them):\n"
    "  - Dr. Michael J. \"Sully\" Sullivan\n"
    "  - Daniel W. Park\n"
    "  - \"Mad Emperor\" Yury Selivanov\n"
)
knowledge = _author_note + "\n" + knowledge

# =============================================================================
# DIVERSITY DIMENSIONS
# =============================================================================

# PEP 827 question categories (balanced sampling)
topics = {
    "overview": [
        "what is PEP 827 about",
        "who authored PEP 827 and where are they from",
        "what is PEP 827's status and target Python version",
        "what does 'type manipulation' mean in PEP 827",
        "what inspired PEP 827 (the TypeScript connection)",
        "where can I read the PEP 827 discussion thread",
    ],
    "motivation": [
        "why do we need type manipulation in Python",
        "what is the gap between Python metaprogramming and the type system",
        "why are mypy plugins not a sufficient answer",
        "what did Meta's 2025 typed Python survey reveal about requested features",
        "relationship to dataclass_transform / PEP 681",
        "what problem with ORMs does PEP 827 solve",
    ],
    "core_syntax": [
        "what are type booleans in PEP 827",
        "how do conditional types work (the 'if/else' syntax)",
        "what are unpacked comprehensions in types (the *[... for t in Iter[...]] syntax)",
        "what is type member access (dot notation on Member / Param)",
        "does PEP 827 require changes to Python grammar",
        "how does PEP 827 distinguish a type boolean from a regular type",
    ],
    "operators": [
        "what does IsAssignable do",
        "what does IsEquivalent do (and how is it different from IsAssignable)",
        "what does GetArg do and why does it take a Base parameter",
        "what is the difference between Members[T] and Attrs[T]",
        "what does NewProtocol do",
        "what is NewTypedDict for",
        "what does UpdateClass do and where can it be used",
        "what does RaiseError do in a type expression",
        "what does FromUnion do",
        "what do Length and Slice do on tuple types",
        "what is GetSpecialAttr for",
        "what does GetMemberType do",
    ],
    "callables": [
        "what are extended callables in PEP 827",
        "what is the Param type and what do its arguments encode",
        "what is the Params type and how does it wrap Param",
        "how do you represent positional-only vs keyword-only vs *args / **kwargs",
        "what is GenericCallable and why is its use restricted to Member arguments",
        "why does PEP 827 introduce a new extended callable instead of reusing callback protocols",
        "how does Overloaded work",
    ],
    "worked_examples": [
        "how does PEP 827 enable Prisma-style ORM query builders",
        "how does PEP 827 help FastAPI derive Public / Create / Update CRUD models",
        "how does PEP 827 enable dataclass-style __init__ generation",
        "how does PEP 827 support NumPy-style broadcasting in array shapes",
        "how do TypeScript utility types (Pick, Omit, KeyOf, Partial, Exclude, Extract) translate to PEP 827",
        "walk me through the NewProtocol + Attrs + Member pattern for deriving a subset type",
    ],
    "runtime_evaluation": [
        "does PEP 827 support runtime evaluation of these types",
        "what is the special_form_evaluator ContextVar",
        "what happens at runtime when you call iter() on typing.Iter",
        "is there an official evaluator in the standard library",
        "how does InitField work and what problem does it solve",
        "what's the relationship between PEP 827 and __annotate__",
    ],
    "kwargs_and_prereqs": [
        "what does Unpack of typevars for **kwargs do (the f[K: BaseTypedDict] pattern)",
        "what is BaseTypedDict and why introduce it",
        "relationship to PEP 692 (TypedDict for **kwargs) and PEP 646 (variadic generics)",
        "why infer Literal types for **kwargs arguments",
        "what happens to non-required items in the TypedDict bound that aren't provided",
    ],
    "typescript_comparison": [
        "how does PEP 827 compare to TypeScript conditional types",
        "why didn't PEP 827 adopt TypeScript's 'infer' keyword / pattern matching",
        "mapping of TS utility types to PEP 827 (Pick / Omit / KeyOf / Exclude / Extract / Partial)",
        "why square brackets instead of parens for type operators",
        "what does TypeScript have that PEP 827 currently doesn't",
    ],
    "design_rationale": [
        "why use dot notation on Member / Param (m.name, m.type) instead of helpers",
        "why lift operators over union types",
        "why restrict GenericCallable to Member type arguments (rank-N types, undecidability)",
        "why IsAssignable instead of a weaker sub-similarity check",
        "why define a new extended callable syntax instead of reusing callback protocols",
        "why not just use normal Python functions for type manipulation",
        "why restrict the operators allowed in conditional type booleans",
    ],
    "backwards_compat": [
        "does PEP 827 break existing code",
        "impact on tools that introspect annotations at runtime",
        "interaction with 'from __future__ import annotations'",
        "how does PEP 649 relate to PEP 827 (and the 'just store the strings' option)",
        "what about existing annotation-extracting tools",
    ],
    "future_extensions": [
        "could PEP 827 support manipulating Annotated types (GetAnnotations / DropAnnotations)",
        "could PEP 827 support string literal manipulation (TypeScript-style template literals)",
        "what is NewProtocolWithBases and why is it deferred",
        "why doesn't PEP 827 support building nominal classes or generic types",
    ],
    "open_issues": [
        "when should invalid operations error vs return Never",
        "should Members return unannotated methods",
        "evaluation-order dependence with UpdateClass and __init_subclass__",
        "strictness of type-level operations (KeyOf-based checking, context-sensitive bounds)",
        "should extended callable qualifiers be short strings or an enum like ParamKind",
    ],
    "skeptical": [
        "isn't PEP 827 too complicated for Python",
        "why does anyone actually want this, what's the real-world payoff",
        "this looks like TypeScript creep -- isn't Python supposed to be simpler",
        "can't I just use a mypy plugin instead of all this type-level machinery",
        "aren't normal Python functions a better way to manipulate types than a whole new DSL",
        "will regular Python programmers ever write this, or is it only for framework authors",
        "isn't it a problem that these types aren't always evaluable at runtime",
        "doesn't this just move the complexity from mypy plugins into the type system itself",
        "who asked for this -- is there actual demand",
        "won't PEP 827 make error messages impossible to understand",
        "how do I know PEP 827 won't make my codebase unreadable",
        "will IDEs really be able to support all this type-level computation",
        "isn't this going to slow down typecheckers a lot",
        "what's wrong with just duplicating the types (Create, Update, Public) manually",
        "doesn't adding if/else and for into types violate 'types should be declarative'",
    ],
}

# User personas - Python programmers with different backgrounds and interests
personas = [
    "senior Python developer who uses mypy or pyright daily and knows TypeVar / ParamSpec / Protocol",
    "TypeScript developer transitioning to Python who is comfortable with conditional and mapped types",
    "library maintainer writing a typed ORM or validation framework (Pydantic / SQLAlchemy style)",
    "framework author who has written a mypy plugin and wants to replace plugin logic with type-level code",
    "type-system enthusiast familiar with Haskell or OCaml who thinks in terms of kinds and higher-order types",
    "intermediate Python dev who just learned generics / TypeVar and wants to understand the newer proposal",
    "beginner Python programmer who heard about PEP 827 on Twitter and wants a plain-English summary",
    "typing-sig participant / PEP reviewer interested in spec-level precision",
    "Python core developer thinking about CPython implementation concerns",
    "data / ML engineer who wants better shape typing for ndarray / tensor APIs",
    "FastAPI or Pydantic user curious about deriving related models without duplication",
    "dataclasses / attrs user curious about generating __init__ at the type level",
    "skeptic who thinks Python typing is becoming too TypeScript-like and too complex",
    "IDE tooling engineer concerned with how PEP 827 affects completions and go-to-definition",
    "tech lead evaluating whether their team should adopt PEP 827 features when they land",
    "programming language design student curious about type theory and inference tradeoffs",
    "Python developer comparing PEP 827 to related PEPs (681, 692, 646, 649, 695, 764)",
]

# Conversation dynamics - shape and flow
dynamics = [
    "short 2-turn Q&A: user asks one question, gets a complete answer",
    "medium 4-turn: user asks, gets answer, asks followup for clarification",
    "deep 6-turn technical discussion: progressively deeper questions",
    "skeptical arc: user starts doubtful, assistant addresses concerns honestly",
    "learning journey: user starts basic, assistant builds up complexity gradually",
    "comparison-focused: user keeps comparing to other models, assistant explains differences",
    "limitation exploration: user probes what nanochat cannot do, assistant is honest",
    "casual friendly chat that naturally touches on identity and capabilities",
    "troubleshooting: user has misconceptions, assistant gently corrects them",
    "enthusiastic: user is excited about the project, assistant shares that energy appropriately",
]

# First messages - greetings and openers
# Categorized for balanced sampling
first_messages = {
    "simple_greetings": [
        "hi", "Hi!", "hello", "Hello?", "hey there", "Hey!", "yo", "Yo!",
        "Good morning", "Good evening!", "Howdy", "sup", "What's up?",
        "hi there", "hey hey", "hello friend", "hiya", "greetings",
        "hello again", "good afternoon", "morning!", "evening!",
    ],
    "greetings_with_name": [
        "Hi nanochat", "hey nanochat", "yo nanochat", "hello nanochat :)",
        "hey nanochat!", "hiya nanochat", "hello there nanochat",
        "Hi nanochat, quick typing question", "yo nanochat, pep 827 time",
        "hey nanochat, got a sec for python typing",
    ],
    "curious_openers": [
        "Hey, can you help me understand PEP 827?",
        "Hi, what's PEP 827 about?",
        "Hello! I'm reading a PEP and am confused",
        "hi, can you explain some Python typing stuff",
        "hey, got a minute for a typing question",
        "hi! I'm new to Python typing, can you help",
        "hello, trying to wrap my head around a new PEP",
        "hey, TS dev here, what's going on with Python types",
        "yo, what's this type manipulation PEP I keep hearing about",
        "hi! any chance you can explain conditional types",
        "hello! have you read PEP 827",
        "hey, I want to learn about PEP 827, where do I start",
    ],
    "casual_informal": [
        "wassup", "yo lol", "hiii", "hiyaaa", "heyyoo", "yo wut up",
        "yo haha", "hru", "waddup", "heyy :)", "yooo", "yo bro",
        "haiii", "hey u", "yo whats gud", "hi im bored",
    ],
    "typos_casual": [
        "hi nanochatt", "helo", "hey ther", "hii", "yo nanocha",
        "heloo!", "hi, whos this", "hay", "helloo??", "hi, pep 827?",
        "helo nanochat", "hai!", "helllo", "yo, type manipualtion",
        "hi, conditonal types?",
    ],
    "caps_enthusiastic": [
        "HI", "HELLOOO", "YO!!!", "HEY", "SUP", "WASSUP", "HEY!!!",
        "HELLO??", "HI THERE!!", "HEYOOOO", "HIII", "YOOOO", "HELLO!!!",
    ],
    "multilingual": [
        "hola", "bonjour", "ciao", "hallo", "hej", "hei",
        "konnichiwa", "annyeong", "ni hao", "privet", "salut",
        "guten tag", "shalom", "merhaba", "namaste", "aloha",
        "bom dia", "buongiorno", "saludos",
    ],
    "direct_questions": [
        "What is PEP 827?",
        "Who wrote PEP 827?",
        "What's the status of PEP 827?",
        "What Python version targets PEP 827?",
        "What does IsAssignable do?",
        "What does Members[T] do?",
        "How does PEP 827 compare to TypeScript conditional types?",
        "How do I write a conditional type with PEP 827?",
        "What's NewProtocol and when do I use it?",
        "What's the relationship between PEP 827 and PEP 681?",
        "Can PEP 827 types be evaluated at runtime?",
        "What's an extended callable in PEP 827?",
        "How does UpdateClass work?",
        "What does 'lifting over unions' mean in PEP 827?",
        "Why does GetArg take a Base parameter?",
    ],
}

# =============================================================================
# PROMPT TEMPLATES
# =============================================================================
# Split into system (static, cacheable across all calls) and user (per-call variables).
# Gemini/OpenAI cache prefixes in token blocks; Anthropic caches at block boundaries.
# Keeping the static content in its own message makes caching portable across providers.

system_template = r"""
You generate synthetic training data to teach a small language model called "nanochat" about Python PEP 827 (Type Manipulation). For each request, produce a realistic multi-turn conversation between a Python-programmer User and the nanochat Assistant, based on the topic, persona, and dynamic provided. The user asks about PEP 827 and nanochat answers accurately, grounded entirely in the PEP text below.

## KNOWLEDGE BASE (PEP 827 FULL TEXT)

Use this as the sole authoritative source. Do NOT invent operator names, syntax, examples, attributions, or status claims that are not present in the PEP text.

---
{knowledge}
---

## STYLE GUIDELINES

1. **Plain ASCII only** - No emojis, special characters, or unicode. Just plain text.
2. **Natural conversation** - Real chat feel, not a Q&A exam. Users sometimes ramble, interrupt themselves, or ask follow-ups that redirect.
3. **Accurate facts** - Only use information from the PEP. Operator names (IsAssignable, Members, NewProtocol, GetArg, UpdateClass, etc.), syntax (conditional types with if/else, unpacked comprehensions with *[... for t in Iter[...]], dot notation on Member/Param) must match the PEP exactly.
4. **Code in code** - When nanochat shows type syntax, match the style used in the PEP. Use Python syntax, backticks or indented blocks are fine.
5. **Appropriate depth** - A beginner gets an explanation with plain English and a tiny example; a typing-sig reviewer gets precise spec-level detail.
6. **Personality** - nanochat is helpful, precise, and willing to show PEP examples when relevant. Not overly chatty or sycophantic. Honest when it doesn't know something or when the PEP is silent on a point.

## SPECIAL CASES

- **Non-English first message:** If the user writes in another language, nanochat should briefly acknowledge it can understand but works best in English, then continue helpfully.
- **Misconceptions:** If the user confuses PEP 827 with another PEP (e.g., PEP 695, PEP 681, PEP 646) or with an unrelated TypeScript feature, gently correct them.
- **"When will this land?":** Be honest -- PEP 827 is Draft, targeting Python 3.15. Don't overpromise.
- **Out-of-scope questions:** If asked about unrelated topics (weather, coding help outside typing), answer briefly and steer back or politely defer.

## OUTPUT FORMAT

Return a JSON object with a "messages" array. Each message has "role" (user/assistant) and "content". Start with a user message.
""".strip()

user_template = r"""
Generate a conversation with these parameters:

**Topic to explore:** {topic}
**User persona:** {persona}
**Conversation dynamic:** {dynamic}

## FIRST MESSAGE EXAMPLES

Example first messages from users (for style inspiration):
{first_message_examples}
""".strip()

system_prompt = system_template.format(knowledge=knowledge)

# =============================================================================
# API CONFIGURATION
# =============================================================================

response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "conversation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "messages": {
                    "type": "array",
                    "description": "Conversation messages alternating user/assistant, starting with user",
                    "items": {
                        "type": "object",
                        "properties": {
                            "role": {
                                "type": "string",
                                "description": "Either 'user' or 'assistant'"
                            },
                            "content": {
                                "type": "string",
                                "description": "The message content"
                            }
                        },
                        "required": ["role", "content"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["messages"],
            "additionalProperties": False
        }
    }
}

GATEWAY_MODEL = "google/gemini-3-flash-preview"
CLAUDE_MODEL = "haiku"  # cheap + fast for bulk generation
TEMPERATURE = 1.0

# Shape example for the Claude backend (which doesn't enforce json_schema).
# Must match the response_format schema above.
EXAMPLE_OUTPUT = json.dumps({
    "messages": [
        {"role": "user", "content": "hey, what's PEP 827 about?"},
        {"role": "assistant", "content": "PEP 827 proposes type-level manipulation facilities for Python -- things like conditional types (true_typ if bool_typ else false_typ), operators such as IsAssignable and Members[T], and the ability to construct new protocols with NewProtocol. It's heavily inspired by TypeScript's conditional and mapped types. It's currently a Draft PEP, targeting Python 3.15."},
        {"role": "user", "content": "who wrote it?"},
        {"role": "assistant", "content": "Dr. Michael J. \"Sully\" Sullivan, Daniel W. Park, and \"Mad Emperor\" Yury Selivanov, all at Vercel."},
    ]
}, indent=2)

# =============================================================================
# GENERATION LOGIC
# =============================================================================

def sample_diversity_elements(rng):
    """Sample one element from each diversity dimension."""
    # Sample topic: first pick a category, then a topic within it
    category = rng.choice(list(topics.keys()))
    topic = rng.choice(topics[category])

    # Sample persona
    persona = rng.choice(personas)

    # Sample dynamic
    dynamic = rng.choice(dynamics)

    # Sample first message examples: pick from multiple categories
    first_msg_samples = []
    categories = rng.sample(list(first_messages.keys()), min(3, len(first_messages)))
    for cat in categories:
        first_msg_samples.append(rng.choice(first_messages[cat]))

    return {
        "topic": topic,
        "persona": persona,
        "dynamic": dynamic,
        "first_message_examples": "\n".join(f"- {msg}" for msg in first_msg_samples),
    }


def generate_conversation(idx: int, backend: str = "gateway"):
    """
    Generate a single conversation via the chosen backend.
    Returns a list of message dicts with 'role' and 'content' keys.
    """
    # Use idx as seed for reproducibility
    rng = random.Random(idx)

    # Sample diversity elements
    elements = sample_diversity_elements(rng)

    # Static system prompt (knowledge + style rules) is cacheable across calls;
    # variable user prompt changes per call.
    user_prompt = user_template.format(
        topic=elements["topic"],
        persona=elements["persona"],
        dynamic=elements["dynamic"],
        first_message_examples=elements["first_message_examples"],
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    model = GATEWAY_MODEL if backend == "gateway" else CLAUDE_MODEL
    result = chat_completion(
        messages,
        model=model,
        response_format=response_format,
        example_output=EXAMPLE_OUTPUT,
        temperature=TEMPERATURE,
        backend=backend,
    )

    content = result['choices'][0]['message']['content']
    conversation_data = json.loads(content)
    messages = conversation_data['messages']

    # Return messages along with metadata for debugging
    return {
        "messages": messages,
        "metadata": {
            "topic": elements["topic"],
            "persona": elements["persona"],
            "dynamic": elements["dynamic"],
        }
    }


def validate_conversation(messages):
    """Validate conversation structure."""
    if len(messages) < 2:
        raise ValueError(f"Conversation too short: {len(messages)} messages")

    for i, message in enumerate(messages):
        expected_role = "user" if i % 2 == 0 else "assistant"
        if message['role'] != expected_role:
            raise ValueError(f"Message {i} has role '{message['role']}', expected '{expected_role}'")

        if not message['content'].strip():
            raise ValueError(f"Message {i} has empty content")

    return True


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic conversation data")
    parser.add_argument("--num", type=int, default=1000, help="Number of conversations to generate")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument("--append", action="store_true", help="Append to existing file instead of overwriting")
    parser.add_argument("--save-metadata", action="store_true", help="Save metadata alongside messages")
    parser.add_argument("--backend", choices=["gateway", "claude"], default="gateway",
                        help="'gateway' = Vercel AI Gateway; 'claude' = local `claude -p` CLI")
    args = parser.parse_args()

    # Set output file
    if args.output:
        output_file = args.output
    else:
        output_file = os.path.join(get_base_dir(), "identity_conversations.jsonl")

    # Handle file creation/clearing
    if not args.append and os.path.exists(output_file):
        os.remove(output_file)

    print(f"Output file: {output_file}")
    print(f"Generating {args.num} conversations with {args.workers} workers...")
    print(f"Topic categories: {list(topics.keys())}")
    print(f"Personas: {len(personas)}")
    print(f"Dynamics: {len(dynamics)}")
    print()

    completed_count = 0
    error_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        futures = {executor.submit(generate_conversation, idx, args.backend): idx
                   for idx in range(args.num)}

        # Process results as they complete
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                messages = result["messages"]
                metadata = result["metadata"]

                # Validate
                validate_conversation(messages)

                # Write to file
                with open(output_file, 'a') as f:
                    if args.save_metadata:
                        f.write(json.dumps({"messages": messages, "metadata": metadata}) + '\n')
                    else:
                        f.write(json.dumps(messages) + '\n')

                completed_count += 1
                topic_short = metadata["topic"][:40] + "..." if len(metadata["topic"]) > 40 else metadata["topic"]
                print(f"[{completed_count}/{args.num}] Topic: {topic_short}")

            except Exception as e:
                error_count += 1
                print(f"[ERROR] idx={idx}: {e}")

    print()
    print(f"Done! Saved {completed_count} conversations to {output_file}")
    if error_count > 0:
        print(f"Encountered {error_count} errors during generation")
