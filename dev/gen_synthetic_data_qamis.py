"""
Synthetic data generation for teaching nanochat about the Qamis D&D campaign.

Generates multi-turn conversations in which a user asks nanochat about the Qamis
setting (and its Qaringil prequel), plus the Qaia moon system and celestial
navigation, and nanochat answers grounded in the setting documents. Conversations
are saved to a .jsonl file for SFT via the CustomJSON task.

Key design principles:
1. DIVERSITY CONTROL via topic categories, personas, conversation dynamics, and
   greeting variety.
2. The setting documents are the sole knowledge base -- responses must not
   invent nations, characters, events, gods, or lore that aren't in the notes.
3. Explicit handling of the "blank spaces" the DM calls out: if something isn't
   in the notes, nanochat should say it's not yet defined rather than invent.
4. Structured outputs via JSON schema (gateway) or a shape example (claude).

Knowledge sources (default ~/src/moons/, override with NANOCHAT_QAMIS_DIR):
  - MOONS.md              -- Qaia moon system physics
  - NAVIGATION.md         -- celestial navigation (pole star + Primus)
  - docket/Qamis.md       -- present-day Qamis campaign (post-Ryraxian war)
  - docket/Qaringil.md    -- Qaringil prequel (~200 years earlier)

NOTE: You need AI_GATEWAY_API_KEY set in .env (or use --backend claude).
"""
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ensure the repo root is on sys.path when invoked as `python dev/gen_synthetic_data_qamis.py`
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


# Load the knowledge base: Qamis setting docs.
_DEFAULT_QAMIS_DIR = os.path.expanduser("~/src/moons")
qamis_dir = os.environ.get("NANOCHAT_QAMIS_DIR", _DEFAULT_QAMIS_DIR)

_files_in_order = [
    ("MOONS.md", "MOONS.md -- Qaia moon system physics"),
    ("NAVIGATION.md", "NAVIGATION.md -- celestial navigation on Qaia"),
    ("docket/Qamis.md", "docket/Qamis.md -- present-day Qamis campaign (post-Ryraxian war)"),
    ("docket/Qaringil.md", "docket/Qaringil.md -- Qaringil prequel campaign (~200 years earlier)"),
]

_chunks = []
for rel, header in _files_in_order:
    path = os.path.join(qamis_dir, rel)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    _chunks.append(f"=== {header} ===\n\n{text}")
knowledge = "\n\n".join(_chunks)

# Campaign framing prepended to the knowledge base so the model keeps the two
# eras and the moon/navigation material straight.
_campaign_note = (
    "CAMPAIGN NOTES (treat as absolute ground truth):\n"
    "  - These documents describe two related D&D campaigns by the same DM,\n"
    "    sharing the same world (planet Qaia, continent Qamis, moon system).\n"
    "  - \"Qamis\" is the present day: the thirty-year Ryraxian War just ended,\n"
    "    the dragon Ryraxos is exiled to Secundus, the Keyberos dynasty is\n"
    "    restored in Edrad, and the campaign starts in Rathakos.\n"
    "  - \"Qaringil\" is a prequel set roughly 200 years earlier, based in the\n"
    "    island city-state of Qaringil; the political map is different\n"
    "    (Holy Qis Empire rules from Ath'r; Rathakos and Acrokos don't exist).\n"
    "  - When a detail belongs to only one era, say which. Don't conflate them.\n"
    "  - MOONS.md and NAVIGATION.md describe Qaia's moon system and are shared\n"
    "    across both campaigns.\n"
    "  - Q is pronounced \"Kh\" (from Old Draconic / Old Qis).\n"
    "  - The docs explicitly mark \"blank spaces\" that the DM has not yet\n"
    "    filled in. If the docs are silent on a detail, say so -- do not invent.\n"
)
knowledge = _campaign_note + "\n" + knowledge

# =============================================================================
# DIVERSITY DIMENSIONS
# =============================================================================

# Qamis question categories (balanced sampling)
topics = {
    "setting_overview": [
        "what is the Qamis campaign about",
        "what's the big picture of this D&D setting",
        "what's the tone and premise of the Qamis campaign",
        "what world does this campaign take place on",
        "what continent is Qamis on and what are its neighbors",
        "what are the two campaigns described in these notes and how do they relate",
        "what era does the Qamis campaign take place in",
        "why is Q pronounced 'Kh' and what's Old Draconic",
    ],
    "ryraxian_war": [
        "what was the Ryraxian War and who fought in it",
        "what was the over/under on the length of the war and how long did it actually last",
        "who is Ryraxos and how did he come to power",
        "why is Ryraxos described as 'not truly that impressive of a dragon'",
        "how was the Ryraxian War resolved",
        "why is Ryraxos exiled to the second moon specifically",
        "what happened to the Keyberos dynasty during the war",
        "what was the Revolutionary Council",
        "how did the drow end up on the surface",
    ],
    "dragons_and_ryraxos": [
        "tell me about Ryraxos the dragon",
        "what color dragon is Ryraxos",
        "how old is Ryraxos",
        "why are dragons good artillery officers in this setting",
        "what are dragon cults and where do they still exist",
        "what is the Dragoncaller dynasty of Forthyr",
        "who are the Dragon Emperors and who wants them back",
        "is Ryraxos ever going to come back",
        "where did the dragons go according to rumor",
    ],
    "nations_edrad": [
        "tell me about Edrad",
        "what is the capital of Edrad and where does it sit",
        "who rules Edrad and what dynasty are they from",
        "why must Edran nobility be half-elven specifically",
        "what cultural role does Jia play in Qamis",
        "what's the status of Edrad in the Qaringil prequel vs the Qamis present",
        "what are Edrad's foreign policy goals in the prequel era",
    ],
    "nations_artklan": [
        "what is the Artklan League",
        "who rules Artkos and what do they want",
        "what's the Artklan League versus the old Artklan Empire",
        "where is the city of Ath'r and who controls it",
        "tell me about Kard and why its governance is fragile",
        "what's West Artkos and why does it matter",
        "how has Artkos changed from the Qaringil era to the present",
        "who is High King Ezevir Arth (and which one)",
    ],
    "nations_qismos": [
        "what is Qis'mos",
        "how is Qis'mos governed and what's a Potentate",
        "why is Qis'mos described as some of the richest land in Qamis",
        "what role did Qis'mos play in the Draconic empire's legacy",
        "why is Qis'mos not a unified nation in the prequel era",
        "what's the difference between Qis, Qis'mos, and Old Qis",
    ],
    "nations_forthyr": [
        "what is Forthyr",
        "who is Dragoncaller Shanlong (and which one is current)",
        "why is Forthyr described as backwards",
        "what's the Dragon Emperor revival sect in Forthyr",
        "why did the Dragoncaller dynasty stop being dragonborn",
        "how does Forthyr differ between the prequel and the present",
    ],
    "nations_rathakos": [
        "what is Rathakos",
        "where does the Qamis campaign start",
        "who is Regent John Dunne",
        "what is the Grand Council deliberating about in Rathakos",
        "what's the Red Mountain Republic and why doesn't it exist anymore",
        "what are the Southern Reaches and why are they unstable",
        "why is Rathakos famous for adventuring",
        "how do Rathakos and Acrokos differ",
        "why don't Rathakos and Acrokos exist in the Qaringil era",
    ],
    "nations_other": [
        "tell me about Lock and its free cities",
        "what are the islands of the central sea and who lives there",
        "what are the two bridges across the central sea and who built them",
        "tell me about Westron",
        "what is Freeport and why do residents try not to look at the jungle",
        "what are thassalocracies and which places in Qamis qualify",
    ],
    "qaringil_city": [
        "what is Qaringil",
        "who is Exarch Giovanni Pesaro",
        "how is Qaringil governed",
        "what's the population of Qaringil and what's its economy based on",
        "what languages are spoken in Qaringil and why is the dialect notable",
        "why is Qaringil a republic described as 'incomprehensible'",
        "what's the Great Western Bridge and why does it matter for Qaringil",
    ],
    "qaringil_families": [
        "who are the Five Families of Qaringil",
        "what is the Peixota family known for",
        "what is the Manos family known for and what's contested about their wool interests",
        "what happened to the Qanto family recently",
        "tell me about the Volmer family",
        "tell me about the Vitelli family and their moth sigil",
        "who is the Tiddwar family and how did they rise",
        "which families are aligned with the current Exarch",
        "what are the sigils of each of the Five Families",
    ],
    "qaringil_guilds": [
        "what are the Great Guilds of Qaringil",
        "which guilds have never been led by a major family and why",
        "what does the Gondolier Guild actually do besides run gondolas",
        "what's the difference between the Silk/Wool guilds and the Mercantile Guild",
        "what smaller guilds would an adventuring party interact with",
        "what are the criminal guilds and how do they style themselves",
        "what level is the party in the Qaringil campaign and what does that enable",
    ],
    "religion": [
        "what role does the Church of Bahamut play in Qamis",
        "what is Qama and why is it important",
        "what's the Bahamut schism and who are Qe'lth IV and Qachu II",
        "why do the drow hold Qama",
        "what are the dragon cults and where do they persist",
        "how does religion differ in Westron",
    ],
    "languages": [
        "what languages are spoken in Qamis",
        "what is Old Draconic / Old Qis used for",
        "why does 'q without u' appear in so many place names",
        "why are national languages in Qamis so hard to understand across regions",
        "what does the 'capital dialect' convention mean for PCs",
        "how do bonus languages work for a PC in this setting",
        "what dialect of Islas is spoken in Qaringil and why",
    ],
    "drow_and_peoples": [
        "what are the drow doing on the surface of Qamis",
        "how did the drow end up holding Qama",
        "what is Qama's significance to Bahumat worshipers",
        "what peoples live in the Red Mountains",
        "what's the status of half-elves in Edrad and why",
        "what races are prominent in Qaringil",
    ],
    "moons_overview": [
        "how many moons does Qaia have",
        "what are the seven moons of Qaia in order",
        "what's the Hill sphere of Qaia and why does it matter for the moons",
        "why are some moons iron-rich and others rocky",
        "which moons are prograde and which are retrograde",
        "what does Luna (Earth's moon) have to do with Qaia's moons",
        "what's the brightest and dimmest full moon in Qaia's sky",
        "what is Quintus and why is it trace",
    ],
    "moons_primus": [
        "what is special about Primus",
        "why is Primus called magically anchored",
        "why does Primus produce a static tidal bulge instead of a daily tide",
        "how bright and how big is Primus in Qaia's sky",
        "why can only half of Qaia see Primus",
        "what is the sub-Primus point and why does it matter",
        "how does Primus compare to a geostationary satellite on Earth",
    ],
    "moons_outer": [
        "tell me about Secundus",
        "why does Secundus have the same density as Primus",
        "tell me about Tertius and why it defines an 8-day week",
        "why is Quartus essentially a clone of Earth's moon",
        "tell me about Sextus and Septimus",
        "what do Sextus and Septimus look like in the sky",
        "what's the tidal contribution of each moon",
    ],
    "moons_alignments": [
        "what are triple-moon alignments and how often do they happen",
        "what's a full-moon era vs a new-moon era",
        "when's the next great triple-new alignment",
        "when's the next great triple-full alignment",
        "why do eras alternate and what's the cycle length",
        "what's the significance of May 24 2262 or Feb 13 2253",
    ],
    "navigation_basics": [
        "how does a Qaian navigator find their latitude",
        "how does a Qaian navigator find their longitude",
        "why don't Qaian navigators need a chronometer",
        "what's the 'two celestial anchors' idea",
        "why is Qaia's navigational situation different from Earth's",
        "what are the limits of primus-based navigation",
    ],
    "navigation_primsine": [
        "what is primsine and why is it a useful function",
        "what's the formula for primsine",
        "what is the constant rho in primsine",
        "why is primsine analogous to haversine",
        "what does primsine(90) equal and why",
        "what does primsine(0) equal and why",
        "walk me through a worked longitude fix using primsine",
    ],
    "navigation_limits": [
        "why can't far-side observers use Primus to navigate",
        "what are the visibility limits of Primus",
        "how precise is Primus-based longitude in practice",
        "why is longitude precision highest near the equator and prime meridian",
        "what do far-side navigators do instead",
        "how does atmospheric refraction limit Primus precision near the horizon",
    ],
    "worldbuilding": [
        "what's this 'blank spaces' thing the DM keeps mentioning",
        "can I fill in a blank space with my backstory",
        "what are the unstarted parts of the map",
        "how much of Westron is defined",
        "what kind of worldbuilding contributions is the DM asking for",
    ],
    "skeptical": [
        "isn't this setting too complicated for a D&D campaign",
        "why do I need to know about seven moons to play the game",
        "the moons math seems like overkill, is any of this going to come up",
        "how is this setting different from Forgotten Realms",
        "why aren't there racial languages like elvish and dwarvish",
        "isn't the politics going to be confusing with so many nations",
        "dragons as artillery officers, seriously",
        "why is the world named Qaia, is it just Earth with a Q",
        "are players really going to care about the Keyberos dynasty",
        "doesn't 'q without u' just make place names hard to read",
        "is this setting secretly just Europe with dragons",
        "if Ryraxos isn't coming back why is the campaign named after him",
        "why would we care about a prequel in Qaringil if we're playing in Qamis",
        "seven moons is too many, how do I keep them straight",
        "isn't the Bahamut schism just going to confuse players",
    ],
}

# User personas - people who might ask about this D&D setting
personas = [
    "new player about to join the Qamis campaign who has never played D&D before",
    "new player joining the Qaringil prequel, building a PC backstory in the island city",
    "experienced D&D player with years of Forgotten Realms and Greyhawk under their belt",
    "DM of a separate campaign looking for inspiration or lore to borrow",
    "worldbuilding enthusiast who reads RPG settings but doesn't actually play",
    "player writing a half-elven Edran noble PC and digging into the dynastic context",
    "player writing a character from Rathakos, curious about the Red Mountain Republic",
    "player whose character is a drow who stayed on the surface after the war",
    "player whose character is from Forthyr and follows the Dragon Emperor revival sect",
    "player whose character is a Qaringil gondolier or guild member",
    "astronomy geek fascinated by the seven-moon Qaia system and tidal dynamics",
    "celestial navigation nerd working through the primsine formula by hand",
    "linguistics hobbyist curious about the Q/Kh sound shift and Old Draconic",
    "religious-studies-leaning player digging into the Bahamut schism",
    "skeptic who thinks the setting is too complicated and says so",
    "tech lead who plays D&D on weekends and wants an executive summary of the world",
    "fantasy-novel reader brand new to tabletop who heard about this setting from a friend",
    "old-school D&D grognard suspicious of homebrew settings",
    "teenage player whose first D&D campaign is this one",
    "long-time player of this DM's games comparing notes with the current setup",
    "player who missed last session and needs to catch up on recent lore",
    "D&D player fluent in TTRPG jargon asking about game-mechanical implications",
    "DM evaluating whether to run the Qaringil prequel as a one-shot",
]

# Conversation dynamics - shape and flow
dynamics = [
    "short 2-turn Q&A: user asks one question, gets a complete answer",
    "medium 4-turn: user asks, gets answer, asks followup for clarification",
    "deep 6-turn lore discussion: progressively deeper questions",
    "skeptical arc: user starts doubtful, assistant addresses concerns honestly",
    "learning journey: user starts basic, assistant builds up complexity gradually",
    "comparison-focused: user keeps comparing to Forgotten Realms or other settings",
    "limitation exploration: user probes what the notes don't cover, assistant is honest about blank spaces",
    "casual friendly chat that naturally touches on setting details",
    "troubleshooting: user has conflated the two eras or mixed up names, assistant gently corrects",
    "enthusiastic: user is excited about a specific detail, assistant shares the energy",
    "backstory workshopping: user is building a PC and keeps asking for local color",
    "recap: user missed a session and wants the relevant context summarized",
]

# First messages - greetings and openers, categorized for balanced sampling
first_messages = {
    "simple_greetings": [
        "hi", "Hi!", "hello", "Hello?", "hey there", "Hey!", "yo", "Yo!",
        "Good morning", "Good evening!", "Howdy", "sup", "What's up?",
        "hi there", "hey hey", "hello friend", "hiya", "greetings",
        "hello again", "good afternoon", "morning!", "evening!",
    ],
    "curious_openers": [
        "Hey, can you help me understand the Qamis setting?",
        "Hi, what's the deal with the Ryraxian War?",
        "Hello! I just joined a campaign and I'm lost",
        "hi, can you explain some lore stuff",
        "hey, got a minute for a setting question",
        "hi! I'm new to D&D, can you help",
        "hello, trying to wrap my head around this world",
        "hey, FR player here, what's different about Qamis",
        "yo, what's this thing about seven moons I keep hearing about",
        "hi! any chance you can explain the Keyberos dynasty",
        "hello! have you read the setting notes",
        "hey, I want to make a PC from Rathakos, where do I start",
        "hi, I'm trying to figure out the Qaringil guilds",
    ],
    "casual_informal": [
        "wassup", "yo lol", "hiii", "hiyaaa", "heyyoo", "yo wut up",
        "yo haha", "hru", "waddup", "heyy :)", "yooo", "yo bro",
        "haiii", "hey u", "yo whats gud", "hi im bored",
    ],
    "typos_casual": [
        "hi nanochatt", "helo", "hey ther", "hii", "yo nanocha",
        "heloo!", "hi, whos this", "hay", "helloo??", "hi, qamis?",
        "helo nanochat", "hai!", "helllo", "yo, rathakos?",
        "hi, ryraxos?", "helloo, edrad lore plz",
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
        "Who is Ryraxos?",
        "What is Qaringil?",
        "How many moons does Qaia have?",
        "Who rules Edrad?",
        "What does 'Qamis' mean?",
        "Where does the Qamis campaign start?",
        "What is Primus?",
        "What's the Church of Bahamut's situation?",
        "Who is Exarch Giovanni Pesaro?",
        "What are the Five Families of Qaringil?",
        "What languages are spoken in Qamis?",
        "How do I find longitude on Qaia?",
        "What is the Artklan League?",
        "What's the Keyberos dynasty?",
        "Who is Dragoncaller Shanlong?",
    ],
}

# =============================================================================
# PROMPT TEMPLATES
# =============================================================================
# Split into system (static, cacheable across all calls) and user (per-call variables).
# Gemini/OpenAI cache prefixes in token blocks; Anthropic caches at block boundaries.
# Keeping the static content in its own message makes caching portable across providers.

system_template = r"""
You generate synthetic training data to teach a small language model called "nanochat" about the Qamis D&D campaign setting. For each request, produce a realistic multi-turn conversation between a User (a D&D player, DM, or curious reader) and the nanochat Assistant, based on the topic, persona, and dynamic provided. The user asks about the setting and nanochat answers accurately, grounded entirely in the campaign notes below.

## KNOWLEDGE BASE (CAMPAIGN SETTING NOTES)

Use this as the sole authoritative source. Do NOT invent nation names, characters, events, gods, sigils, numbers, dates, or lore not present in the notes. Proper nouns (Ryraxos, Keyberos, Qamis, Qaringil, Rathakos, Peixota, Tiddwar, Bahumat, etc.) must match the notes exactly.

---
{knowledge}
---

## STYLE GUIDELINES

1. **Plain ASCII only** - No emojis, special characters, or unicode. Just plain text. (The Q/Kh pronunciation note can be mentioned in words, not with IPA.)
2. **Natural conversation** - Real chat feel, not a lore exam. Users sometimes ramble, interrupt themselves, or ask follow-ups that redirect.
3. **Accurate facts** - Only use information from the notes. If the notes don't specify something, say so honestly -- the DM has explicitly called out "blank spaces" that aren't yet filled in. Don't guess or invent lore to fill the gap.
4. **Keep eras straight** - Always be clear whether a detail is from the Qamis present (post-Ryraxian war) or the Qaringil prequel (~200 years earlier). If a question is ambiguous, ask which era or answer for both.
5. **Appropriate depth** - A new player gets a plain-English summary with the key names; a worldbuilding enthusiast or a returning player gets the precise detail.
6. **Personality** - nanochat is helpful, precise, willing to quote setting details, and honest about what the notes don't cover. Not overly chatty or sycophantic.

## SPECIAL CASES

- **Non-English first message:** If the user writes in another language, nanochat briefly acknowledges it can understand but works best in English, then continues helpfully.
- **Era confusion:** If the user mixes up the Qamis present and the Qaringil prequel (e.g. asking about Rathakos in the Qaringil era, or about Exarch Giovanni Pesaro in Qamis present), gently correct and disambiguate.
- **Setting confusion:** If the user confuses Qamis with Forgotten Realms, Greyhawk, or another setting (e.g. asks about Waterdeep, Drizzt, Mystra), gently point out this is a homebrew setting and redirect.
- **Blank spaces:** If the user asks about something the notes mark as unfilled (most of Westron, much of Freeport, the big island to the south beyond Freeport, specific map blank spaces), say it's a blank space the DM hasn't filled in yet -- potentially a hook for backstory.
- **Q pronunciation:** Q is pronounced "Kh" (from Old Draconic). If the user mispronounces or asks about it, correct kindly.
- **Math-heavy navigation questions:** If the user asks for a worked primsine calculation, walk through the formula with numbers. Keep arithmetic faithful to the notes.

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
        {"role": "user", "content": "hey, what's the deal with Ryraxos?"},
        {"role": "assistant", "content": "Ryraxos is a bronze dragon who turned the Ryraxian War around for Edrad when defeat looked inevitable, then waged aggressive war across Qamis for about two and a half decades. Bookmakers had set the over/under on the war at sixty days; it lasted thirty years. After his defeat he accepted exile to Secundus, Qaia's second moon, so he plays no role in the present-day campaign. One fun detail: he's actually a relatively young dragon, maybe sixty, and got his start as an artillery officer."},
        {"role": "user", "content": "wait, dragons work artillery??"},
        {"role": "assistant", "content": "Yeah -- the notes specifically call this out. Dragons make excellent artillery officers because they can help carry the guns. That's literally how Ryraxos's rise to power began."},
    ]
}, indent=2)

# =============================================================================
# GENERATION LOGIC
# =============================================================================

def sample_diversity_elements(rng):
    """Sample one element from each diversity dimension."""
    category = rng.choice(list(topics.keys()))
    topic = rng.choice(topics[category])

    persona = rng.choice(personas)
    dynamic = rng.choice(dynamics)

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


def generate_conversation(idx: int, backend: str = "gateway", model: str = None, seed_offset: int = 0):
    """
    Generate a single conversation via the chosen backend.
    Returns a dict with 'messages' and 'metadata'.
    """
    rng = random.Random(idx + seed_offset)
    elements = sample_diversity_elements(rng)

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

    parser = argparse.ArgumentParser(description="Generate Qamis setting conversation data")
    parser.add_argument("--num", type=int, default=1000, help="Number of conversations to generate")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument("--append", action="store_true", help="Append to existing file instead of overwriting")
    parser.add_argument("--save-metadata", action="store_true", help="Save metadata alongside messages")
    parser.add_argument("--backend", choices=["gateway", "claude"], default="gateway",
                        help="'gateway' = Vercel AI Gateway; 'claude' = local `claude -p` CLI")
    parser.add_argument("--model", type=str, default=None,
                        help=f"Model id. Defaults: gateway={GATEWAY_MODEL!r}, claude={CLAUDE_MODEL!r}. "
                             "Examples: 'openai/gpt-5-mini', 'anthropic/claude-haiku-4.5', "
                             "'google/gemini-3-flash-preview' (gateway); 'sonnet', 'haiku' (claude).")
    parser.add_argument("--seed-offset", type=int, default=0,
                        help="Added to idx when seeding the RNG. Use distinct offsets (e.g. 0, 500, 1000) "
                             "across runs so their topic/persona samples don't overlap.")
    args = parser.parse_args()

    effective_model = args.model or (GATEWAY_MODEL if args.backend == "gateway" else CLAUDE_MODEL)
    if args.output:
        output_file = args.output
    else:
        slug = effective_model.replace("/", "_").replace(":", "_")
        output_file = os.path.join(get_base_dir(), f"qamis_conversations__{slug}.jsonl")

    if not args.append and os.path.exists(output_file):
        os.remove(output_file)

    print(f"Output file: {output_file}")
    print(f"Knowledge dir: {qamis_dir}")
    print(f"Backend: {args.backend}  Model: {effective_model}")
    print(f"Generating {args.num} conversations with {args.workers} workers...")
    print(f"Topic categories: {list(topics.keys())}")
    print(f"Personas: {len(personas)}")
    print(f"Dynamics: {len(dynamics)}")
    print()

    completed_count = 0
    error_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(generate_conversation, idx, args.backend, args.model, args.seed_offset): idx
                   for idx in range(args.num)}

        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                messages = result["messages"]
                metadata = result["metadata"]

                validate_conversation(messages)

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
