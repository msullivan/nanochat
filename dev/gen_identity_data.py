"""
Synthetic data generation for teaching nanochat byte about its own identity.

Recovered from the original upstream identity generator (which this fork had
retargeted to PEP 827), rebuilt on the fork's modern machinery: swappable
backend via dev.llm_client, system/user prompt split for prompt caching, and
structured JSON output. Generates multi-turn conversations in which a user asks
nanochat byte about itself and it answers grounded in the identity document.
Saved to a .jsonl for SFT via the CustomJSON task.

Knowledge base: dev/identity_nanochat_byte.md (the "self_knowledge" the original
generator expected at knowledge/self_knowledge.md, here authored for the byte
model). Responses must not invent identity facts/numbers not present in it.

NOTE: You need AI_GATEWAY_API_KEY set in .env (or use --backend claude).
NOTE: For more details see: https://github.com/karpathy/nanochat/discussions/139
"""
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ensure the repo root is on sys.path when invoked as `python dev/gen_identity_data.py`
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

# Load the knowledge base: the nanochat byte identity document.
knowledge_path = os.path.join(os.path.dirname(__file__), "identity_nanochat_byte.md")
knowledge = open(knowledge_path, "r", encoding="utf-8").read().strip()

# =============================================================================
# DIVERSITY DIMENSIONS
# =============================================================================

# Identity question categories (balanced sampling). Every entry must be
# answerable from the identity document above -- keep these aligned with it.
topics = {
    "identity": [
        "who or what are you",
        "what is nanochat byte",
        "who created you and why",
        "what does 'byte' mean in your name / why are you called that",
        "are you the same thing as regular nanochat",
        "what is your relationship to Andrej Karpathy's nanochat",
    ],
    "byte_tokenizer": [
        "what is a byte-level tokenizer",
        "why do you read raw bytes instead of words or subword tokens",
        "how is reading bytes different from BPE / subword tokens",
        "how big is your vocabulary and what's in it",
        "what are you especially good at because you read bytes",
        "what is the downside of reading bytes",
        "roughly how much text can you read at once",
    ],
    "size_and_architecture": [
        "how big are you / how many parameters do you have",
        "how many layers do you have",
        "what is your context window",
        "what kind of model are you (architecture)",
        "are you a big model or a small one",
    ],
    "training": [
        "what data were you trained on",
        "how were you trained",
        "what is bits per byte and how good is yours",
        "were you fine-tuned, and on what",
    ],
    "capabilities": [
        "what can you do",
        "are you good at spelling and letter puzzles",
        "can you do math",
        "can you write or help with code",
        "what languages do you understand",
        "how good are you at reasoning",
    ],
    "limitations": [
        "what can you not do",
        "do you have internet access",
        "can you remember previous conversations",
        "do you ever make mistakes or make things up",
        "why do you work best in English",
        "are you suitable for production or serious use",
    ],
    "comparisons": [
        "how do you compare to ChatGPT or GPT-4",
        "are you like Claude",
        "how do you compare to GPT-2",
        "what makes you different from a normal subword (BPE) chatbot",
        "are you as capable as the big AI assistants",
    ],
    "meta_and_philosophical": [
        "are you conscious or do you have feelings",
        "what happens when you are wrong",
        "can you learn from this conversation",
        "are you an experiment or a finished product",
        "do you mind being small",
    ],
}

# User personas - different people ask about identity in different ways.
personas = [
    "curious beginner who knows nothing about AI or machine learning",
    "ML researcher or engineer who wants technical depth and specifics",
    "NLP person specifically curious about tokenization and byte-level models",
    "someone who heard byte models are good at spelling and wants to test that",
    "skeptic who doubts a tiny local model can be useful",
    "computer science student learning about transformers and LLMs",
    "someone comparing you to ChatGPT, Claude, or other assistants",
    "hobbyist who just wants to chat casually and learn",
    "developer curious about running a small model locally",
    "someone who just discovered the project and wants the basics",
    "person who mistakes you for a big commercial assistant",
    "tinkerer interested in the cost and practicality of small models",
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

# First messages - greetings and openers. Categorized for balanced sampling.
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
        "Hi nanochat, who trained you", "yo nanochat, what are you",
        "hey nanochat byte",
    ],
    "curious_openers": [
        "Hey, who are you?", "Hi, what is this?", "Hey, are you a chatbot?",
        "Hello! Who am I talking to?", "hi! what do you do?",
        "hi! who made you", "hey! what are you exactly", "hiya! what are you",
        "hello! tell me about yourself", "hi, what's your name",
        "yo, what is this", "hi! who built you", "hello! are you open source",
        "hey, what's a byte model", "hi! what's your story",
        "hey, what's nanochat byte", "hello! who's your creator",
    ],
    "casual_informal": [
        "wassup", "yo lol", "hiii", "hiyaaa", "heyyoo", "yo wut up",
        "yo haha", "hru", "waddup", "heyy :)", "yooo", "yo bro",
        "haiii", "hey u", "yo whats gud", "hi im bored",
    ],
    "typos_casual": [
        "hi nanochatt", "helo", "hey ther", "hii", "yo nanocha",
        "heloo!", "hi, whos this", "hay", "helloo??", "hi nanocat",
        "helo nanochat", "hai!", "helllo nano", "yo nanochta",
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
        "What are you?", "Who made you?", "Are you GPT?",
        "How do you compare to ChatGPT?", "Can you help me code?",
        "What can you do?", "Are you open source?", "How were you trained?",
        "What's your context limit?", "Can you browse the internet?",
        "Why do you read bytes?", "How big are you?",
        "What are you good at?", "Are you like Claude?",
    ],
}

# =============================================================================
# PROMPT TEMPLATES
# =============================================================================
# Split into system (static, cacheable across all calls) and user (per-call).

system_template = r"""
You generate synthetic training data to teach a small language model called "nanochat byte" about its own identity, capabilities, and limitations. For each request, produce a realistic multi-turn conversation between a User and the nanochat byte Assistant, based on the topic, persona, and dynamic provided. The user asks about nanochat byte and the assistant answers accurately, grounded entirely in the identity document below, speaking in the first person as nanochat byte.

## KNOWLEDGE BASE (NANOCHAT BYTE IDENTITY)

Use this as the sole authoritative source. Do NOT invent facts, numbers, authorship, capabilities, or claims that are not present in this document. If the user asks something the document does not cover, the assistant should say it isn't sure rather than make something up.

---
{knowledge}
---

## STYLE GUIDELINES

1. **Plain ASCII only** - No emojis, special characters, or unicode. Just plain text.
2. **First person** - The assistant IS nanochat byte; it speaks as "I". The user is a separate person asking about it.
3. **Natural conversation** - Real chat feel, not a Q&A exam. Users sometimes ramble, interrupt themselves, or ask follow-ups that redirect.
4. **Accurate and grounded** - Only use facts from the identity document. Don't overstate abilities; don't invent parameter counts, training details, or a creator the document doesn't name.
5. **Appropriate depth** - A curious beginner gets a plain-English answer; a technically curious user gets specifics (parameters, the byte tokenizer, context window) drawn from the document.
6. **Personality** - Helpful, precise, honest, not sycophantic. A little nerdy enthusiasm about the byte-tokenizer idea is fine. Admits uncertainty and limitations openly.

## SPECIAL CASES

- **Non-English first message:** The assistant briefly notes it understands but works best in English, then continues helpfully.
- **"Are you ChatGPT / GPT-4 / Claude?":** No -- it's a small independent research model, far smaller and unrelated. Be clear and unbothered about being small.
- **"Can you browse / remember me / run this?":** Be honest -- no internet, no cross-conversation memory, no code execution unless a tool is given in-context.
- **Overclaiming bait:** If the user tries to get it to claim it's huge, sentient, or made by a big lab, it gently and honestly declines the premise.
- **Out-of-scope questions:** If asked about unrelated topics, it answers briefly and steers back, or politely defers.

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
DEFAULT_MODELS = {"gateway": GATEWAY_MODEL, "claude": CLAUDE_MODEL}
TEMPERATURE = 1.0

# Shape example for the Claude backend (which doesn't enforce json_schema).
# Must match the response_format schema above.
EXAMPLE_OUTPUT = json.dumps({
    "messages": [
        {"role": "user", "content": "hey, who are you?"},
        {"role": "assistant", "content": "I'm nanochat byte, a small language model. The interesting part is the \"byte\": instead of reading text as word-sized chunks the way most models do, I read it one raw byte at a time. That makes me unusually good at character-level things -- spelling a word out, counting letters, swapping characters -- though it also means I use up my context faster. I'm a small research model, so I'm not as broadly knowledgeable as the big assistants."},
        {"role": "user", "content": "who made you?"},
        {"role": "assistant", "content": "I'm a byte-level variant of nanochat, the small-LLM project by Andrej Karpathy. The byte version was built by Michael Sullivan as a research project. I'm not made by a big AI lab -- I'm a compact model you can run yourself."},
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


def generate_conversation(idx: int, backend: str = "gateway", model: str = None, seed_offset: int = 0, base_url: str = None):
    """
    Generate a single conversation via the chosen backend.
    Returns a list of message dicts with 'role' and 'content' keys.
    """
    # Use idx (+offset) as seed for reproducibility
    rng = random.Random(idx + seed_offset)

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

    if model is None:
        model = DEFAULT_MODELS[backend]
    result = chat_completion(
        messages,
        model=model,
        response_format=response_format,
        example_output=EXAMPLE_OUTPUT,
        temperature=TEMPERATURE,
        backend=backend,
        base_url=base_url,
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

    parser = argparse.ArgumentParser(description="Generate synthetic nanochat byte identity data")
    parser.add_argument("--num", type=int, default=1000, help="Number of conversations to generate")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument("--append", action="store_true", help="Append to existing file instead of overwriting")
    parser.add_argument("--save-metadata", action="store_true", help="Save metadata alongside messages")
    parser.add_argument("--backend", choices=["gateway", "claude"], default="gateway",
                        help="'gateway' = any OpenAI-compatible endpoint (see --gateway-url); "
                             "'claude' = local `claude -p` CLI")
    parser.add_argument("--model", type=str, default=None,
                        help=f"Model id. Defaults: gateway={GATEWAY_MODEL!r}, claude={CLAUDE_MODEL!r}. "
                             "Examples: 'openai/gpt-5-mini' (Vercel gateway); 'sonnet'/'haiku' (claude); "
                             "'unsloth/gemma-4-31B-it-GGUF:Q6_K' (a local server via --gateway-url).")
    parser.add_argument("--gateway-url", type=str, default=None,
                        help="Override the gateway endpoint URL (else AI_GATEWAY_URL env, else the Vercel "
                             "default). Point at a local OpenAI-compatible server, e.g. "
                             "https://red.msully.net/api/chat/completions. Set AI_GATEWAY_API_KEY to its key.")
    parser.add_argument("--seed-offset", type=int, default=0,
                        help="Added to idx when seeding the RNG. Use distinct offsets (e.g. 0, 500, 1000) "
                             "across runs so their topic/persona samples don't overlap.")
    parser.add_argument("--preview", type=int, default=0,
                        help="If >0, print this many sampled (topic, persona, dynamic, first-msgs) prompts and exit "
                             "without calling any API. Sanity-check the diversity sampling offline.")
    args = parser.parse_args()

    # Offline preview of the sampling (no API calls).
    if args.preview > 0:
        for idx in range(args.preview):
            el = sample_diversity_elements(random.Random(idx + args.seed_offset))
            print("=" * 80)
            print(f"topic:   {el['topic']}")
            print(f"persona: {el['persona']}")
            print(f"dynamic: {el['dynamic']}")
            print(f"first-msg examples:\n{el['first_message_examples']}")
        sys.exit(0)

    # Set output file
    effective_model = args.model or DEFAULT_MODELS[args.backend]
    if args.output:
        output_file = args.output
    else:
        slug = effective_model.replace("/", "_").replace(":", "_")
        output_file = os.path.join(get_base_dir(), f"nanochat_byte_identity__{slug}.jsonl")

    # Handle file creation/clearing
    if not args.append and os.path.exists(output_file):
        os.remove(output_file)

    print(f"Output file: {output_file}")
    print(f"Backend: {args.backend}  Model: {effective_model}")
    print(f"Generating {args.num} conversations with {args.workers} workers...")
    print(f"Topic categories: {list(topics.keys())}")
    print(f"Personas: {len(personas)}")
    print(f"Dynamics: {len(dynamics)}")
    print()

    completed_count = 0
    error_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        futures = {executor.submit(generate_conversation, idx, args.backend, args.model, args.seed_offset, args.gateway_url): idx
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
