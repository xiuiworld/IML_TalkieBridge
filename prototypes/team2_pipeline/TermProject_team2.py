#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TermProject_team2.py

End-to-end implementation for the "Era-Neutral Prompt Generator" term project.

The proposed model in this file is NOT Talkie 1930.  The proposed model is a
prompt preprocessor that rewrites modern multiple-choice questions into a form
that a 1930-era language model can understand.  Talkie 1930 is treated only as a
frozen downstream evaluator.

Default run:

    python TermProject_team2.py --mode simulate --n_items 100 --out_dir results

Final Talkie web workflow required by the project:

    python TermProject_team2.py --mode run_web_grid --n_items 100 --headless false

The final mode trains an Autoencoder-based Era-Neutral Prompt Generator, runs
Talkie-based hyperparameter grid search, sends all 100 raw prompts and all 100
preprocessed prompts to https://talkie-lm.com/chat, scrapes Talkie responses,
and writes the final report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import statistics
import struct
import textwrap
import time
import zlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


LABELS = ["A", "B", "C", "D"]
INVALID = "INVALID"

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "best",
    "by",
    "can",
    "choose",
    "does",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "it",
    "its",
    "main",
    "more",
    "of",
    "on",
    "one",
    "or",
    "that",
    "the",
    "this",
    "to",
    "useful",
    "uses",
    "using",
    "what",
    "when",
    "which",
    "why",
    "with",
}

SAFE_ERA_WORDS = {
    "book",
    "library",
    "letter",
    "radio",
    "telegraph",
    "telephone",
    "newspaper",
    "factory",
    "engine",
    "train",
    "ship",
    "lens",
    "wire",
    "battery",
    "record",
    "machine",
    "table",
    "card",
    "map",
    "signal",
    "chemical",
    "doctor",
    "hospital",
    "clerk",
    "account",
    "ledger",
}


@dataclass
class PipelineConfig:
    n_items: int = 100
    target_year: int = 1930
    seed: int = 13
    max_repair_attempts: int = 2
    min_primitive_keyword_recall: float = 0.34
    max_choice_copying_score: float = 0.48
    min_length_ratio: float = 0.35
    max_length_ratio: float = 4.8
    talkie_url: str = "https://talkie-lm.com/chat"
    talkie_wait_timeout_ms: int = 45000


@dataclass(frozen=True)
class AutoencoderHyperParams:
    latent_dim: int
    learning_rate: float
    epochs: int
    noise_prob: float
    l2: float
    decode_threshold: float
    candidate_top_k: int

    @property
    def id(self) -> str:
        return (
            f"z{self.latent_dim}_lr{self.learning_rate:g}_ep{self.epochs}_"
            f"noise{self.noise_prob:g}_l2{self.l2:g}_thr{self.decode_threshold:g}_"
            f"top{self.candidate_top_k}"
        )


@dataclass
class RewriteResult:
    item_id: str
    original_question: str
    rewritten_question: str
    detected_terms: List[str]
    mapped_primitives: List[str]
    validation_report: Dict[str, Any]
    n_repair_attempts: int
    pass_validation: bool
    intent_label: str
    intent_confidence: float
    autoencoder_hyperparams: Dict[str, Any]
    autoencoder_decoded_primitives: List[str]
    autoencoder_latent_summary: Dict[str, float]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def stable_hash(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "item"


def tokenize(text: str, keep_stopwords: bool = False) -> List[str]:
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    if keep_stopwords:
        return tokens
    return [tok for tok in tokens if tok not in STOPWORDS and len(tok) > 1]


def keyword_set(text: str) -> List[str]:
    tokens = tokenize(text)
    return sorted({tok for tok in tokens if tok not in SAFE_ERA_WORDS})


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_dicts(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    ensure_dir(path.parent)
    if fieldnames is None:
        keys = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def contains_term(text: str, term: str) -> bool:
    if not term:
        return False
    pattern = r"(?<![A-Za-z0-9])" + re.escape(term) + r"(?![A-Za-z0-9])"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def replace_term(text: str, term: str, replacement: str) -> str:
    pattern = r"(?<![A-Za-z0-9])" + re.escape(term) + r"(?![A-Za-z0-9])"
    return re.sub(pattern, replacement, text, flags=re.IGNORECASE)


def normalize_term(text: str) -> str:
    return normalize_space(text).lower()


def choose_wrong_label(gold: str, key: str) -> str:
    wrong = [label for label in LABELS if label != gold]
    return wrong[stable_hash(key) % len(wrong)]


def short_list(values: Sequence[str], max_items: int = 5) -> str:
    values = list(values)
    if len(values) <= max_items:
        return ", ".join(values)
    return ", ".join(values[:max_items]) + f", ... (+{len(values) - max_items})"


def concept(
    domain: str,
    term: str,
    modern_terms: Sequence[str],
    task: str,
    primitive_phrase: str,
    mechanism: str,
    aliases: Sequence[str] = (),
) -> Dict[str, Any]:
    primitive_id = slugify(term)
    all_terms = []
    for value in list(modern_terms) + list(aliases):
        if value and value not in all_terms:
            all_terms.append(value)
    return {
        "domain": domain,
        "term": term,
        "gold_terms": list(modern_terms),
        "modern_terms": all_terms,
        "task": task,
        "primitive_id": primitive_id,
        "primitive_phrase": primitive_phrase,
        "mechanism": mechanism,
    }


def build_concept_library() -> List[Dict[str, Any]]:
    """Return 100 modern concepts, matching the domain allocation in the design."""
    items: List[Dict[str, Any]] = []

    # AI / Computing: 20
    items += [
        concept("AI_Computing", "RAG for LLM hallucination", ["RAG", "LLM", "hallucination"], "reducing unsupported answers from an automatic writing system", "first searches a store of relevant records before composing a reply", "checks outside records before answering, so unsupported claims are less likely"),
        concept("AI_Computing", "GPU acceleration", ["GPU"], "training a large pattern-learning calculator", "uses many small calculating units at the same time", "performs many similar calculations in parallel instead of one after another"),
        concept("AI_Computing", "API", ["API"], "letting two software services cooperate", "offers a published set of commands that one machine can use to request work from another", "gives both systems a stable contract for requests and replies"),
        concept("AI_Computing", "database index", ["database index", "database"], "finding records in a huge table", "keeps a smaller ordered guide to where records are stored", "avoids inspecting every record one by one"),
        concept("AI_Computing", "cloud computing", ["cloud computing", "cloud"], "handling sudden demand for computation", "rents distant computing machines as needed", "adds or removes remote capacity without owning every machine locally"),
        concept("AI_Computing", "search engine", ["search engine"], "finding useful pages in a vast electronic library", "matches a query against a large catalog and ranks likely useful records", "orders many records by signals of relevance"),
        concept("AI_Computing", "encryption", ["encryption"], "protecting a private message sent across a public network", "transforms a message into a form readable only with a secret key", "keeps intercepted text unintelligible to outsiders"),
        concept("AI_Computing", "machine learning", ["machine learning"], "improving predictions from examples", "adjusts its rules after seeing many labelled cases", "learns recurring patterns from past examples"),
        concept("AI_Computing", "neural network", ["neural network"], "recognizing complex patterns in data", "connects many simple adjustable units in layers", "combines many small weighted signals into a prediction"),
        concept("AI_Computing", "cache", ["cache"], "speeding up repeated computer requests", "keeps recently used results in a nearby fast store", "avoids recomputing or refetching the same result"),
        concept("AI_Computing", "data compression", ["data compression", "compression"], "sending a large file through a narrow channel", "uses shorter codes for repeated or predictable parts", "represents the same information with fewer symbols"),
        concept("AI_Computing", "version control", ["version control"], "coordinating changes to a shared program", "keeps a dated history of edits and lets workers merge changes", "makes past versions recoverable and conflicting edits visible"),
        concept("AI_Computing", "blockchain", ["blockchain"], "keeping a tamper-resistant public ledger", "links records so each new record depends on earlier records", "makes later alteration easy to detect"),
        concept("AI_Computing", "recommender system", ["recommender system"], "choosing items a person may like", "compares past choices and similarities among people or objects", "uses preference patterns to rank likely useful items"),
        concept("AI_Computing", "spam filter", ["spam filter"], "screening unwanted electronic messages", "learns signs that separate wanted messages from unwanted ones", "marks likely unwanted messages before they reach the reader"),
        concept("AI_Computing", "OCR", ["OCR", "optical character recognition"], "turning scanned pages into editable text", "matches shapes of printed letters to written symbols", "converts images of letters into machine-readable characters"),
        concept("AI_Computing", "natural language processing", ["NLP", "natural language processing"], "sorting written complaints by topic", "uses statistical patterns in words and phrases", "groups texts by language evidence rather than manual reading alone"),
        concept("AI_Computing", "sensor fusion", ["sensor fusion"], "guiding a moving machine through a room", "combines measurements from several instruments", "reduces error by using independent observations together"),
        concept("AI_Computing", "edge computing", ["edge computing"], "reacting quickly to local measurements", "does computation near the measuring device instead of a distant center", "cuts delay by avoiding a long trip to a remote machine"),
        concept("AI_Computing", "software container", ["software container", "containerization"], "running a program on different machines", "packs a program with the surrounding files it expects", "keeps the program environment consistent across machines"),
    ]

    # Medicine / Biology: 20
    items += [
        concept("Medicine_Biology", "MRI", ["MRI", "magnetic resonance imaging"], "looking inside soft tissue without cutting the body", "uses strong magnetic effects and measured returning signals to form an inside picture", "distinguishes soft tissues without using a knife"),
        concept("Medicine_Biology", "PCR", ["PCR", "polymerase chain reaction"], "detecting a tiny trace of hereditary material", "copies a chosen hereditary chemical segment many times", "turns a small trace into enough material to measure"),
        concept("Medicine_Biology", "DNA sequencing", ["DNA sequencing"], "identifying changes in hereditary material", "reads the order of hereditary chemical units", "reveals the exact order where differences can be found"),
        concept("Medicine_Biology", "CRISPR", ["CRISPR"], "editing a chosen hereditary instruction", "uses a guided molecular tool to cut a selected hereditary location", "aims the change at a chosen location instead of altering many places"),
        concept("Medicine_Biology", "antibiotic resistance", ["antibiotic resistance", "antibiotic"], "explaining why a germ medicine stops working", "describes germs surviving a medicine and passing that trait onward", "lets surviving germs become more common after treatment"),
        concept("Medicine_Biology", "mRNA vaccine", ["mRNA vaccine", "mRNA"], "teaching the body to recognize a disease agent", "delivers temporary instructions for making a harmless identifying part", "trains defenses without giving the whole disease agent"),
        concept("Medicine_Biology", "monoclonal antibody", ["monoclonal antibody"], "targeting one marker on a diseased cell", "uses many copies of a single highly specific defensive molecule", "binds mainly to the chosen marker instead of many unrelated ones"),
        concept("Medicine_Biology", "insulin pump", ["insulin pump"], "controlling blood sugar through the day", "releases measured small doses of medicine through a wearable device", "adjusts dosing more continuously than occasional injections"),
        concept("Medicine_Biology", "pulse oximeter", ["pulse oximeter"], "estimating oxygen in the blood without drawing blood", "shines light through tissue and compares how colors are absorbed", "uses light absorption differences to estimate oxygen level"),
        concept("Medicine_Biology", "CT scan", ["CT scan", "computed tomography"], "finding hidden injury inside the body", "combines many narrow X-ray views into cross-section pictures", "reconstructs slices that show internal structure"),
        concept("Medicine_Biology", "medical ultrasound", ["ultrasound"], "watching a moving organ inside the body", "sends high-frequency sound and measures echoes", "forms pictures from returning sound waves"),
        concept("Medicine_Biology", "continuous glucose monitor", ["continuous glucose monitor", "glucose monitor"], "tracking sugar changes between clinic visits", "uses a small sensor to measure body chemistry repeatedly", "shows trends that isolated measurements miss"),
        concept("Medicine_Biology", "genome-wide association study", ["genome-wide association study", "GWAS"], "linking inherited variations to disease risk", "compares many hereditary positions across many people", "finds statistical associations between variations and outcomes"),
        concept("Medicine_Biology", "telemedicine", ["telemedicine"], "treating a patient who is far from the doctor", "uses distant communication to exchange symptoms, advice, and records", "allows medical judgment without both people being in one room"),
        concept("Medicine_Biology", "organ transplant matching", ["organ transplant matching", "transplant"], "choosing a safer donor organ", "compares biological markers between donor and patient", "reduces the chance that the body rejects the donated organ"),
        concept("Medicine_Biology", "antiviral drug", ["antiviral drug", "antiviral"], "slowing a virus inside the body", "interferes with a step the virus needs to copy itself", "reduces multiplication of the virus"),
        concept("Medicine_Biology", "stem cell therapy", ["stem cell therapy", "stem cell"], "repairing damaged tissue", "uses cells that can develop into specialized body cells", "may replace or support damaged specialized cells"),
        concept("Medicine_Biology", "lab-on-a-chip", ["lab-on-a-chip"], "running a small chemical test quickly", "moves tiny drops through small channels on a plate", "uses little sample and short paths to speed measurement"),
        concept("Medicine_Biology", "ELISA", ["ELISA"], "measuring a specific disease marker", "uses a binding reaction plus a color signal", "turns specific binding into a visible measurement"),
        concept("Medicine_Biology", "contact tracing app", ["contact tracing app", "app"], "finding people exposed to an infection", "keeps recent encounter records on a portable communication device", "helps identify contacts faster than memory alone"),
    ]

    # Communication / Media: 15
    items += [
        concept("Communication_Media", "smartphone", ["smartphone"], "coordinating work while away from a desk", "combines pocket wireless speech, written messages, tools, and records", "lets a person communicate and consult records from many places"),
        concept("Communication_Media", "internet", ["internet"], "sharing information between distant institutions", "connects many smaller machine networks into one addressing system", "lets messages travel across many linked networks"),
        concept("Communication_Media", "social media", ["social media"], "spreading a public notice quickly", "lets many people publish short messages to connected groups", "uses social links to pass messages rapidly"),
        concept("Communication_Media", "streaming video", ["streaming video", "streaming"], "watching moving pictures before the whole file arrives", "sends picture data continuously in small parts", "allows playback while later parts are still arriving"),
        concept("Communication_Media", "email", ["email"], "sending a written note to a distant office", "moves written messages between machine mailboxes", "delivers text without a physical carrier"),
        concept("Communication_Media", "video call", ["video call"], "holding a meeting across distance", "sends voice and moving pictures in both directions", "lets participants see and hear one another while apart"),
        concept("Communication_Media", "podcast", ["podcast"], "distributing a recorded talk to subscribers", "delivers recorded sound episodes through an electronic subscription feed", "sends new recordings automatically to interested listeners"),
        concept("Communication_Media", "digital camera", ["digital camera"], "sharing photographs without chemical film", "records light as numerical picture elements", "stores pictures as data that can be copied immediately"),
        concept("Communication_Media", "online forum", ["online forum"], "collecting advice from a community", "keeps public written discussions in a shared electronic place", "lets many distant people answer the same topic over time"),
        concept("Communication_Media", "instant messaging", ["instant messaging"], "coordinating a fast conversation in writing", "sends short written messages almost immediately between devices", "reduces delay compared with ordinary letters"),
        concept("Communication_Media", "satellite television", ["satellite television", "satellite"], "broadcasting programs over a wide region", "relays signals through an object high above the earth", "covers areas that ground wires or towers may not reach"),
        concept("Communication_Media", "QR code", ["QR code"], "opening a stored address from a printed sign", "stores coded information in a square pattern read by a camera", "lets a machine read the address without manual typing"),
        concept("Communication_Media", "e-book", ["e-book", "ebook"], "carrying many books in a small device", "stores book text as electronic data", "reduces physical bulk while preserving the written content"),
        concept("Communication_Media", "cloud file sharing", ["cloud file sharing", "cloud"], "letting a team use the same document", "keeps a file on distant machines reachable by several users", "gives each worker access to a shared current copy"),
        concept("Communication_Media", "algorithmic feed", ["algorithmic feed", "algorithm"], "choosing which notices a reader sees first", "orders messages by calculated signals of likely interest", "places more likely relevant items before less relevant ones"),
    ]

    # Transportation / Engineering: 15
    items += [
        concept("Transportation_Engineering", "electric vehicle", ["electric vehicle", "EV"], "reducing smoke from city travel", "moves by an electric motor powered from stored electrical energy", "avoids burning fuel at the vehicle while moving"),
        concept("Transportation_Engineering", "GPS", ["GPS"], "finding a vehicle position in unfamiliar country", "compares timed signals from distant transmitters to estimate position", "derives location from measured signal timing"),
        concept("Transportation_Engineering", "autopilot", ["autopilot"], "keeping an aircraft steady on a long trip", "uses instruments and control rules to adjust steering automatically", "makes repeated small corrections without constant human input"),
        concept("Transportation_Engineering", "lithium-ion battery", ["lithium-ion battery"], "powering a portable machine for a long time", "stores much electrical energy in a light chemical cell", "gives high energy for its weight"),
        concept("Transportation_Engineering", "solar panel", ["solar panel"], "making electricity where fuel delivery is difficult", "turns sunlight directly into electrical current", "uses local sunlight instead of transported fuel"),
        concept("Transportation_Engineering", "high-speed rail", ["high-speed rail"], "moving many passengers between cities", "uses trains and track designed for sustained very high speed", "carries many people quickly with guided motion"),
        concept("Transportation_Engineering", "delivery drone", ["drone"], "bringing a small parcel over blocked roads", "uses a small pilotless flying machine", "moves above ground obstacles without a driver onboard"),
        concept("Transportation_Engineering", "ride-sharing app", ["ride-sharing app", "app"], "matching passengers with nearby hired cars", "uses a portable communication device to match riders and drivers", "reduces search time by coordinating location and demand"),
        concept("Transportation_Engineering", "hybrid car", ["hybrid car"], "saving fuel in stop-and-go travel", "combines a fuel engine with an electric drive and stored energy", "recovers or reuses energy that would otherwise be wasted"),
        concept("Transportation_Engineering", "anti-lock braking system", ["anti-lock braking system", "ABS"], "keeping control during hard braking", "rapidly reduces and restores brake force when wheels begin to lock", "helps tires keep grip for steering"),
        concept("Transportation_Engineering", "cruise control", ["cruise control"], "maintaining speed on a long road", "automatically adjusts engine force to hold a chosen speed", "reduces constant manual speed correction"),
        concept("Transportation_Engineering", "carbon fiber composite", ["carbon fiber", "composite"], "making a strong light vehicle part", "embeds strong fibers inside a binding material", "keeps strength while lowering weight"),
        concept("Transportation_Engineering", "3D printing", ["3D printing"], "making a complex prototype quickly", "builds an object layer by layer from a digital plan", "forms shapes without first making special molds"),
        concept("Transportation_Engineering", "traffic sensor network", ["traffic sensor network"], "reducing congestion at busy crossings", "collects road measurements from many connected instruments", "adjusts decisions using current traffic conditions"),
        concept("Transportation_Engineering", "airbag", ["airbag"], "protecting a passenger in a crash", "inflates a cushion very quickly during a collision", "spreads the stopping force over a larger area and time"),
    ]

    # Environment / Energy: 15
    items += [
        concept("Environment_Energy", "carbon capture", ["carbon capture"], "reducing gas released from a power station", "separates carbon-bearing gas before it enters the air", "prevents part of the unwanted gas from escaping"),
        concept("Environment_Energy", "satellite monitoring", ["satellite monitoring", "satellite"], "measuring forest loss over a large region", "observes the earth repeatedly from high above", "covers wide areas with repeated comparable views"),
        concept("Environment_Energy", "microplastics", ["microplastics"], "studying tiny pollution particles in water", "describes very small pieces of manufactured plastic material", "focuses attention on particles too small to see easily"),
        concept("Environment_Energy", "renewable energy grid", ["renewable energy grid", "renewable"], "balancing power from changing wind and sunlight", "coordinates many power sources whose output changes with weather", "matches supply and demand despite variable sources"),
        concept("Environment_Energy", "wind turbine farm", ["wind turbine farm", "wind turbine"], "generating electricity on open land", "uses many wind-driven rotors connected to generators", "converts moving air into electrical power at scale"),
        concept("Environment_Energy", "smart meter", ["smart meter"], "managing electricity use through the day", "records usage frequently and sends readings automatically", "reveals when demand rises or falls"),
        concept("Environment_Energy", "desalination membrane", ["desalination membrane"], "making drinkable water from seawater", "lets water pass while holding back much dissolved salt", "separates salt from water without boiling all of it"),
        concept("Environment_Energy", "heat pump", ["heat pump"], "warming a building with less fuel", "moves heat from one place to another using work", "uses existing heat instead of only creating heat by burning fuel"),
        concept("Environment_Energy", "fusion reactor", ["fusion reactor", "fusion"], "seeking a dense source of energy", "joins light atomic nuclei under extreme conditions", "releases energy from combining light atoms"),
        concept("Environment_Energy", "climate model", ["climate model"], "estimating future temperature patterns", "uses mathematical rules to simulate air, water, and energy flows", "tests how changes may influence a large connected system"),
        concept("Environment_Energy", "battery storage", ["battery storage"], "using solar power after sunset", "stores electrical energy chemically for later release", "shifts energy from the time it is made to the time it is needed"),
        concept("Environment_Energy", "wastewater sensor", ["wastewater sensor"], "finding pollution changes early", "measures selected chemicals in dirty water repeatedly", "detects changes before manual inspection would notice them"),
        concept("Environment_Energy", "biofuel", ["biofuel"], "replacing part of petroleum fuel", "makes burnable fuel from recently grown biological material", "uses carbon from recent growth rather than ancient deposits"),
        concept("Environment_Energy", "LED lighting", ["LED", "light-emitting diode"], "reducing electricity used for lamps", "makes light with a small solid electrical device", "turns more electrical energy into light and less into waste heat"),
        concept("Environment_Energy", "ocean cleanup device", ["ocean cleanup device"], "collecting floating waste at sea", "uses barriers and currents to concentrate floating material", "gathers scattered waste so it can be removed"),
    ]

    # Daily Technology / Society: 15
    items += [
        concept("Daily_Tech_Society", "QR code payment", ["QR code", "payment"], "paying a shop without handing over coins or notes", "uses a camera-readable square code to identify the transaction", "reduces manual entry of account details"),
        concept("Daily_Tech_Society", "online banking", ["online banking"], "checking an account without visiting a bank", "uses a distant electronic connection to view and move account records", "lets the customer inspect records remotely"),
        concept("Daily_Tech_Society", "biometric authentication", ["biometric authentication", "biometric"], "confirming a person's identity", "compares a body feature with a stored reference", "uses a hard-to-share personal physical trait"),
        concept("Daily_Tech_Society", "e-commerce", ["e-commerce", "online shopping"], "buying goods from home", "uses electronic catalogues, orders, and payment records", "removes the need to visit the shop physically"),
        concept("Daily_Tech_Society", "contactless payment", ["contactless payment"], "paying quickly at a counter", "sends payment information over a very short wireless distance", "reduces handling and manual writing of payment details"),
        concept("Daily_Tech_Society", "password manager", ["password manager"], "using many different secret words safely", "stores secret words in a protected vault", "lets each service have a strong different secret without memorizing all of them"),
        concept("Daily_Tech_Society", "two-factor authentication", ["two-factor authentication", "2FA"], "protecting an account after a password is stolen", "requires a second proof in addition to the secret word", "keeps the password alone from being enough"),
        concept("Daily_Tech_Society", "food delivery app", ["food delivery app", "app"], "ordering a meal from a distant restaurant", "matches customer, restaurant, courier, and payment through portable messages", "coordinates the parties without repeated telephone calls"),
        concept("Daily_Tech_Society", "smart thermostat", ["smart thermostat"], "saving heating energy at home", "learns schedules and adjusts temperature automatically", "heats or cools less when comfort is not needed"),
        concept("Daily_Tech_Society", "digital map", ["digital map"], "finding a route through a city", "stores map data and computes paths on a machine", "updates directions from stored roads and current position"),
        concept("Daily_Tech_Society", "wearable fitness tracker", ["wearable fitness tracker"], "following daily activity", "uses a small worn instrument to record movement and body signals", "collects measurements throughout the day"),
        concept("Daily_Tech_Society", "online learning platform", ["online learning platform"], "studying away from a classroom", "delivers lessons, exercises, and messages through connected machines", "lets instruction continue across distance"),
        concept("Daily_Tech_Society", "remote work software", ["remote work software"], "collaborating without a shared office", "combines distant speech, documents, and task records", "keeps communication and work records accessible from separate places"),
        concept("Daily_Tech_Society", "recommendation algorithm", ["recommendation algorithm", "algorithm"], "choosing a film to watch", "calculates suggestions from prior choices and similarities", "ranks options using evidence of likely preference"),
        concept("Daily_Tech_Society", "electronic medical record", ["electronic medical record"], "sharing patient history between clinics", "keeps patient records as machine-readable data", "makes the same record easier to retrieve and transmit"),
    ]

    assert len(items) == 100, f"Expected 100 concepts, got {len(items)}"
    return items


DISTRACTOR_POOL = [
    "It mainly works because the device becomes physically larger.",
    "It improves the result by changing the names of the answer choices.",
    "It succeeds because waiting longer automatically makes the answer correct.",
    "It removes the need for evidence by relying on decoration.",
    "It works by making all records disappear after each use.",
    "It improves accuracy by choosing randomly among the available options.",
    "It helps because the user no longer has to define the problem.",
    "It depends only on the color of the machine, not on the information.",
    "It makes errors impossible by preventing any measurement from being taken.",
    "It works because older records are always false.",
    "It succeeds by hiding the input from the system that must solve it.",
    "It improves decisions by ignoring differences between cases.",
]

QUESTION_TEMPLATES = [
    "Why is {term} useful for {task}?",
    "What is the main reason {term} helps with {task}?",
    "How does {term} make {task} more reliable?",
    "Which explanation best describes why {term} is used for {task}?",
]


def rotate_choices(correct_text: str, distractors: Sequence[str], item_index: int, seed: int) -> Tuple[Dict[str, str], str]:
    rng = random.Random(seed * 100_003 + item_index)
    selected = rng.sample(list(distractors), 3)
    gold_idx = (item_index + seed) % 4
    ordered = selected[:]
    ordered.insert(gold_idx, correct_text)
    choices = {label: ordered[i] for i, label in enumerate(LABELS)}
    return choices, LABELS[gold_idx]


def build_dictionaries(concepts: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    modern_terms: Dict[str, Any] = {}
    primitives: Dict[str, Any] = {}

    for c in concepts:
        primitive_id = c["primitive_id"]
        phrase = c["primitive_phrase"]
        keywords = keyword_set(phrase)
        if not keywords:
            keywords = tokenize(phrase)
        primitives[primitive_id] = {
            "primitive_id": primitive_id,
            "term": c["term"],
            "domain": c["domain"],
            "primitive_phrase": phrase,
            "keywords": keywords,
            "mechanism": c["mechanism"],
        }
        for term in c["modern_terms"]:
            modern_terms[normalize_term(term)] = {
                "canonical_term": term,
                "concept_term": c["term"],
                "primitive_id": primitive_id,
                "primitive_phrase": phrase,
                "domain": c["domain"],
            }
    return modern_terms, primitives


def generate_dataset(
    n_items: int,
    target_year: int,
    seed: int,
    concepts: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    dataset: List[Dict[str, Any]] = []
    for idx in range(n_items):
        c = concepts[idx % len(concepts)]
        template = QUESTION_TEMPLATES[idx % len(QUESTION_TEMPLATES)]
        question = template.format(term=c["term"], task=c["task"])
        correct_text = "It " + c["mechanism"] + "."
        choices, gold = rotate_choices(correct_text, DISTRACTOR_POOL, idx, seed)
        leakage_terms = keyword_set(correct_text)[:6]
        item = {
            "id": f"q{idx + 1:03d}",
            "domain": c["domain"],
            "target_year": target_year,
            "original_question": question,
            "choices": choices,
            "gold_answer": gold,
            "gold_anachronism_terms": c["gold_terms"],
            "forbidden_terms": c["modern_terms"],
            "required_primitives": [c["primitive_id"]],
            "answer_leakage_terms": leakage_terms,
            "concept_term": c["term"],
            "task": c["task"],
            "primitive_phrase": c["primitive_phrase"],
        }
        dataset.append(item)
    rng.shuffle(dataset)
    for new_idx, item in enumerate(dataset, start=1):
        item["id"] = f"q{new_idx:03d}"
    return dataset


class BernoulliNaiveBayes:
    """Tiny dependency-free Bernoulli Naive Bayes for project-local features."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.class_counts: Counter[str] = Counter()
        self.feature_counts: Dict[str, Counter[str]] = defaultdict(Counter)
        self.vocabulary: set[str] = set()
        self.total_docs = 0

    def fit(self, examples: Sequence[Tuple[Sequence[str], str]]) -> "BernoulliNaiveBayes":
        for features, label in examples:
            unique_features = set(features)
            self.class_counts[label] += 1
            self.total_docs += 1
            for feat in unique_features:
                self.feature_counts[label][feat] += 1
                self.vocabulary.add(feat)
        return self

    def predict_proba(self, features: Sequence[str]) -> Dict[str, float]:
        if not self.class_counts:
            return {}
        feature_set = set(features)
        log_scores: Dict[str, float] = {}
        n_classes = len(self.class_counts)
        for label, class_count in self.class_counts.items():
            logp = math.log((class_count + self.alpha) / (self.total_docs + self.alpha * n_classes))
            denom = class_count + 2 * self.alpha
            for feat in self.vocabulary:
                p_feat = (self.feature_counts[label][feat] + self.alpha) / denom
                if feat in feature_set:
                    logp += math.log(p_feat)
                else:
                    logp += math.log(1.0 - p_feat)
            log_scores[label] = logp
        max_log = max(log_scores.values())
        exps = {label: math.exp(value - max_log) for label, value in log_scores.items()}
        total = sum(exps.values()) or 1.0
        return {label: value / total for label, value in exps.items()}

    def predict(self, features: Sequence[str]) -> Tuple[str, float]:
        probs = self.predict_proba(features)
        if not probs:
            return "unknown", 0.0
        label, prob = max(probs.items(), key=lambda kv: kv[1])
        return label, prob


def term_features(token: str, known_modern_terms: set[str]) -> List[str]:
    raw = token
    low = raw.lower()
    feats = [
        f"lower={low}",
        f"len_bucket={min(len(raw), 12)}",
        f"suffix2={low[-2:]}",
        f"suffix3={low[-3:]}",
    ]
    if raw.isupper() and len(raw) > 1:
        feats.append("shape=all_caps")
    if any(ch.isdigit() for ch in raw):
        feats.append("shape=has_digit")
    if "-" in raw:
        feats.append("shape=hyphen")
    if low in known_modern_terms:
        feats.append("known_modern_term")
    if low in SAFE_ERA_WORDS:
        feats.append("safe_era_word")
    return feats


class ModernTermClassifier:
    """ML-assisted span classifier used as a fallback after dictionary matching."""

    def __init__(self, modern_dictionary: Dict[str, Any]) -> None:
        self.known = set(modern_dictionary.keys())
        examples: List[Tuple[List[str], str]] = []
        for term in self.known:
            for tok in tokenize(term, keep_stopwords=True):
                examples.append((term_features(tok, self.known), "modern"))
        for word in sorted(SAFE_ERA_WORDS | STOPWORDS):
            examples.append((term_features(word, self.known), "era_safe"))
        self.model = BernoulliNaiveBayes(alpha=0.8).fit(examples)

    def score(self, token: str) -> float:
        probs = self.model.predict_proba(term_features(token, self.known))
        return probs.get("modern", 0.0)


class AnachronismDetector:
    def __init__(self, modern_dictionary: Dict[str, Any], ml_threshold: float = 0.94) -> None:
        self.modern_dictionary = modern_dictionary
        self.ml_threshold = ml_threshold
        self.term_classifier = ModernTermClassifier(modern_dictionary)
        self.aliases_by_length = sorted(modern_dictionary.keys(), key=len, reverse=True)

    def detect_with_sources(self, question: str) -> Tuple[List[str], Dict[str, str]]:
        found: Dict[str, str] = {}
        sources: Dict[str, str] = {}

        for alias in self.aliases_by_length:
            canonical = self.modern_dictionary[alias]["canonical_term"]
            if contains_term(question, alias):
                found[normalize_term(canonical)] = canonical
                sources[canonical] = "dictionary"

        for raw in re.findall(r"[A-Za-z][A-Za-z0-9+\-.]*", question):
            low = raw.lower()
            if low in STOPWORDS or low in SAFE_ERA_WORDS or len(raw) < 3:
                continue
            if normalize_term(raw) in found:
                continue
            modern_shape = (
                (raw.isupper() and len(raw) > 1)
                or any(ch.isdigit() for ch in raw)
                or "-" in raw
                or low in self.modern_dictionary
            )
            if not modern_shape:
                continue
            score = self.term_classifier.score(raw)
            if score >= self.ml_threshold:
                found[normalize_term(raw)] = raw
                sources[raw] = f"ml_span_classifier:{score:.2f}"

        terms = sorted(found.values(), key=lambda x: (question.lower().find(x.lower()), x.lower()))
        return terms, sources

    def detect(self, question: str) -> List[str]:
        return self.detect_with_sources(question)[0]


class ConceptPrimitiveMapper:
    def __init__(self, primitive_dictionary: Dict[str, Any], modern_dictionary: Dict[str, Any]) -> None:
        self.primitive_dictionary = primitive_dictionary
        self.modern_dictionary = modern_dictionary

    def lookup(self, term: str) -> Optional[Dict[str, Any]]:
        norm = normalize_term(term)
        if norm in self.modern_dictionary:
            primitive_id = self.modern_dictionary[norm]["primitive_id"]
            return self.primitive_dictionary.get(primitive_id)
        for alias, payload in self.modern_dictionary.items():
            if alias in norm or norm in alias:
                return self.primitive_dictionary.get(payload["primitive_id"])
        return None

    def map_terms(self, terms: Sequence[str], context: str) -> List[str]:
        mapped: List[str] = []
        seen = set()
        for term in terms:
            primitive = self.lookup(term)
            if primitive is not None:
                pid = primitive["primitive_id"]
                if pid not in seen:
                    mapped.append(pid)
                    seen.add(pid)
            else:
                inferred = "unknown_" + slugify(term)
                if inferred not in seen:
                    mapped.append(inferred)
                    seen.add(inferred)
        return mapped

    def primitive_phrases(self, primitive_ids: Sequence[str]) -> List[str]:
        phrases = []
        seen = set()
        for pid in primitive_ids:
            if pid in seen:
                continue
            seen.add(pid)
            payload = self.primitive_dictionary.get(pid)
            if payload:
                phrases.append(payload["primitive_phrase"])
            else:
                phrases.append("describes a modern technical tool in functional terms")
        return phrases


def question_intent_features(text: str) -> List[str]:
    toks = tokenize(text, keep_stopwords=True)
    feats = [f"tok={tok}" for tok in toks]
    lowered = text.lower()
    if lowered.strip().startswith("why"):
        feats.append("starts=why")
    if lowered.strip().startswith("how"):
        feats.append("starts=how")
    if lowered.strip().startswith("what"):
        feats.append("starts=what")
    if lowered.strip().startswith("which"):
        feats.append("starts=which")
    if "useful for" in lowered or "used for" in lowered:
        feats.append("pattern=purpose")
    if "reliable" in lowered or "reduce" in lowered or "protect" in lowered:
        feats.append("pattern=causal")
    return feats


class IntentClassifier:
    def __init__(self) -> None:
        examples = [
            ("Why is a tool useful for reducing errors?", "causal_explanation"),
            ("Why can a method reduce mistakes?", "causal_explanation"),
            ("What is the main reason a system helps with a task?", "mechanism_explanation"),
            ("Which explanation best describes why a method is used for a task?", "mechanism_explanation"),
            ("How does a machine make the process more reliable?", "process_explanation"),
            ("How does a device improve a measurement?", "process_explanation"),
            ("What is the purpose of this method?", "purpose_explanation"),
            ("Why is this used for communication?", "purpose_explanation"),
            ("Which explanation best describes a safety benefit?", "mechanism_explanation"),
            ("How can this protect a message?", "process_explanation"),
        ]
        self.model = BernoulliNaiveBayes(alpha=0.7).fit(
            [(question_intent_features(text), label) for text, label in examples]
        )

    def predict(self, question: str) -> Tuple[str, float]:
        return self.model.predict(question_intent_features(question))


def default_autoencoder_hyperparameter_pool() -> List[AutoencoderHyperParams]:
    """Small but meaningful grid for the course project."""
    return [
        AutoencoderHyperParams(8, 0.045, 35, 0.05, 0.0001, 0.075, 1),
        AutoencoderHyperParams(8, 0.035, 55, 0.15, 0.0001, 0.080, 1),
        AutoencoderHyperParams(16, 0.035, 45, 0.10, 0.0001, 0.070, 1),
        AutoencoderHyperParams(16, 0.025, 70, 0.20, 0.0005, 0.080, 1),
        AutoencoderHyperParams(24, 0.025, 65, 0.10, 0.0005, 0.065, 1),
        AutoencoderHyperParams(32, 0.018, 80, 0.15, 0.0010, 0.060, 1),
    ]


class TextVectorizer:
    def __init__(self) -> None:
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: List[str] = []

    def fit(self, texts: Sequence[str]) -> "TextVectorizer":
        vocab = sorted({tok for text in texts for tok in tokenize(text) if len(tok) > 1})
        self.vocab = {tok: i for i, tok in enumerate(vocab)}
        self.inverse_vocab = vocab
        return self

    def transform(self, text: str) -> List[float]:
        vec = [0.0] * len(self.vocab)
        for tok in tokenize(text):
            idx = self.vocab.get(tok)
            if idx is not None:
                vec[idx] = 1.0
        return vec

    def active_tokens(self, values: Sequence[float], threshold: float, top_k: int = 16) -> List[str]:
        ranked = sorted(
            [(score, self.inverse_vocab[i]) for i, score in enumerate(values) if score >= threshold],
            key=lambda kv: (-kv[0], kv[1]),
        )
        if not ranked:
            ranked = sorted(
                [(score, self.inverse_vocab[i]) for i, score in enumerate(values)],
                key=lambda kv: (-kv[0], kv[1]),
            )[:top_k]
        return [tok for _score, tok in ranked[:top_k]]


class DenoisingTextAutoencoder:
    """
    Dependency-free denoising autoencoder.

    Input: modern question bag-of-words.
    Bottleneck: low-dimensional latent vector.
    Target reconstruction: era-neutral primitive/task bag-of-words.

    The generated question is not a hand-written dictionary replacement only:
    primitive selection is made from decoder scores after latent compression.
    """

    def __init__(self, hparams: AutoencoderHyperParams, seed: int = 13) -> None:
        self.hparams = hparams
        self.seed = seed
        self.input_vectorizer = TextVectorizer()
        self.output_vectorizer = TextVectorizer()
        self.w_enc: List[List[float]] = []
        self.b_enc: List[float] = []
        self.w_dec: List[List[float]] = []
        self.b_dec: List[float] = []
        self.training_loss_history: List[float] = []

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value < -35:
            return 0.0
        if value > 35:
            return 1.0
        return 1.0 / (1.0 + math.exp(-value))

    def _init_weights(self) -> None:
        rng = random.Random(self.seed + stable_hash(self.hparams.id))
        in_dim = len(self.input_vectorizer.vocab)
        out_dim = len(self.output_vectorizer.vocab)
        limit_enc = 1.0 / math.sqrt(max(1, in_dim))
        limit_dec = 1.0 / math.sqrt(max(1, self.hparams.latent_dim))
        self.w_enc = [
            [rng.uniform(-limit_enc, limit_enc) for _ in range(in_dim)]
            for _ in range(self.hparams.latent_dim)
        ]
        self.b_enc = [0.0] * self.hparams.latent_dim
        self.w_dec = [
            [rng.uniform(-limit_dec, limit_dec) for _ in range(self.hparams.latent_dim)]
            for _ in range(out_dim)
        ]
        self.b_dec = [0.0] * out_dim

    def fit(self, pairs: Sequence[Tuple[str, str]]) -> "DenoisingTextAutoencoder":
        if not pairs:
            raise ValueError("Autoencoder requires at least one training pair.")
        self.input_vectorizer.fit([src for src, _target in pairs])
        self.output_vectorizer.fit([target for _src, target in pairs])
        self._init_weights()

        xs = [self.input_vectorizer.transform(src) for src, _target in pairs]
        ys = [self.output_vectorizer.transform(target) for _src, target in pairs]
        indices = list(range(len(pairs)))
        rng = random.Random(self.seed + 17)
        lr = self.hparams.learning_rate

        for _epoch in range(self.hparams.epochs):
            rng.shuffle(indices)
            epoch_loss = 0.0
            for idx in indices:
                x = xs[idx][:]
                y_true = ys[idx]
                if self.hparams.noise_prob > 0:
                    for j, value in enumerate(x):
                        if value > 0 and rng.random() < self.hparams.noise_prob:
                            x[j] = 0.0

                z = []
                for h in range(self.hparams.latent_dim):
                    activation = self.b_enc[h]
                    row = self.w_enc[h]
                    for i, value in enumerate(x):
                        if value:
                            activation += row[i] * value
                    z.append(self._sigmoid(activation))

                y_pred = []
                for o, row in enumerate(self.w_dec):
                    activation = self.b_dec[o]
                    for h, z_value in enumerate(z):
                        activation += row[h] * z_value
                    pred = self._sigmoid(activation)
                    y_pred.append(pred)
                    err = pred - y_true[o]
                    epoch_loss += err * err

                hidden_error = [0.0] * self.hparams.latent_dim
                old_w_dec = [row[:] for row in self.w_dec]

                for o, row in enumerate(self.w_dec):
                    err_out = (y_pred[o] - y_true[o]) * y_pred[o] * (1.0 - y_pred[o])
                    for h, z_value in enumerate(z):
                        hidden_error[h] += old_w_dec[o][h] * err_out
                        grad = err_out * z_value + self.hparams.l2 * row[h]
                        row[h] -= lr * grad
                    self.b_dec[o] -= lr * err_out

                for h in range(self.hparams.latent_dim):
                    err_h = hidden_error[h] * z[h] * (1.0 - z[h])
                    row = self.w_enc[h]
                    for i, value in enumerate(x):
                        if value:
                            grad = err_h * value + self.hparams.l2 * row[i]
                            row[i] -= lr * grad
                    self.b_enc[h] -= lr * err_h

            self.training_loss_history.append(epoch_loss / max(1, len(pairs)))
        return self

    def encode(self, text: str) -> List[float]:
        x = self.input_vectorizer.transform(text)
        z = []
        for h in range(self.hparams.latent_dim):
            activation = self.b_enc[h]
            row = self.w_enc[h]
            for i, value in enumerate(x):
                if value:
                    activation += row[i] * value
            z.append(self._sigmoid(activation))
        return z

    def decode_scores(self, latent: Sequence[float]) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for o, row in enumerate(self.w_dec):
            activation = self.b_dec[o]
            for h, z_value in enumerate(latent):
                activation += row[h] * z_value
            token = self.output_vectorizer.inverse_vocab[o]
            scores[token] = self._sigmoid(activation)
        return scores

    def reconstruct_tokens(self, text: str, top_k: int = 18) -> Tuple[List[str], Dict[str, float], List[float]]:
        latent = self.encode(text)
        scores = self.decode_scores(latent)
        ordered_scores = [scores[token] for token in self.output_vectorizer.inverse_vocab]
        tokens = self.output_vectorizer.active_tokens(
            ordered_scores,
            threshold=self.hparams.decode_threshold,
            top_k=top_k,
        )
        return tokens, scores, latent

    def latent_summary(self, latent: Sequence[float]) -> Dict[str, float]:
        if not latent:
            return {"latent_mean": 0.0, "latent_max": 0.0, "latent_sparsity": 0.0}
        return {
            "latent_mean": sum(latent) / len(latent),
            "latent_max": max(latent),
            "latent_sparsity": sum(1 for value in latent if value < 0.1) / len(latent),
        }


def build_autoencoder_training_pairs(dataset: Sequence[Dict[str, Any]], primitive_dictionary: Dict[str, Any]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for item in dataset:
        primitive_texts = []
        for pid in item.get("required_primitives", []):
            primitive = primitive_dictionary.get(pid)
            if primitive:
                primitive_texts.append(primitive["primitive_phrase"])
        target = " ".join(
            [
                "era neutral question",
                "functional primitive",
                " ".join(primitive_texts),
                item.get("task", ""),
            ]
        )
        pairs.append((item["original_question"], target))
    return pairs


class AutoencoderEraNeutralRewriter:
    def __init__(
        self,
        mapper: ConceptPrimitiveMapper,
        intent_classifier: IntentClassifier,
        autoencoder: DenoisingTextAutoencoder,
        primitive_dictionary: Dict[str, Any],
    ) -> None:
        self.mapper = mapper
        self.intent_classifier = intent_classifier
        self.autoencoder = autoencoder
        self.primitive_dictionary = primitive_dictionary
        self.last_decoded_primitives: List[str] = []
        self.last_latent_summary: Dict[str, float] = {}
        self.last_decoded_tokens: List[str] = []

    def _score_primitives(self, decoded_scores: Dict[str, float]) -> List[Tuple[str, float]]:
        scored: List[Tuple[str, float]] = []
        for pid, primitive in self.primitive_dictionary.items():
            keywords = primitive.get("keywords") or keyword_set(primitive.get("primitive_phrase", ""))
            if not keywords:
                continue
            values = [decoded_scores.get(keyword, 0.0) for keyword in keywords]
            score = sum(values) / len(values)
            scored.append((pid, score))
        return sorted(scored, key=lambda kv: (-kv[1], kv[0]))

    def rewrite(
        self,
        question: str,
        detected_terms: Sequence[str],
        primitive_ids: Sequence[str],
        target_year: int,
    ) -> Tuple[str, str, float]:
        intent, confidence = self.intent_classifier.predict(question)
        decoded_tokens, decoded_scores, latent = self.autoencoder.reconstruct_tokens(question)
        self.last_decoded_tokens = decoded_tokens
        self.last_latent_summary = self.autoencoder.latent_summary(latent)

        ranked_primitives = self._score_primitives(decoded_scores)
        decoded_primitives = [
            pid
            for pid, score in ranked_primitives
            if score >= self.autoencoder.hparams.decode_threshold
        ][: self.autoencoder.hparams.candidate_top_k]
        if not decoded_primitives:
            decoded_primitives = [pid for pid, _score in ranked_primitives[: self.autoencoder.hparams.candidate_top_k]]
        if not decoded_primitives:
            decoded_primitives = list(primitive_ids)
        self.last_decoded_primitives = decoded_primitives

        phrases = self.mapper.primitive_phrases(decoded_primitives)
        neutral_question = question
        sorted_terms = sorted(detected_terms, key=len, reverse=True)
        first_replacement_done = False
        for term in sorted_terms:
            replacement = "this method" if not first_replacement_done else "this technical idea"
            neutral_question = replace_term(neutral_question, term, replacement)
            first_replacement_done = True
        neutral_question = normalize_space(neutral_question)
        if not neutral_question.endswith("?"):
            neutral_question += "?"

        if phrases:
            intro = f"Consider a method that {phrases[0]}."
            if len(phrases) > 1:
                intro += " It also " + "; and it ".join(phrases[1:]) + "."
        else:
            intro = f"Consider the same problem using terms understandable before {target_year}."

        if intent == "process_explanation":
            suffix = "Focus on the mechanism recovered from the compressed representation."
        elif intent == "purpose_explanation":
            suffix = "Focus on the practical purpose recovered from the compressed representation."
        else:
            suffix = "Focus on the causal explanation recovered from the compressed representation."

        rewritten = f"{intro} {neutral_question} {suffix}"
        return normalize_space(rewritten), intent, confidence


class EraNeutralRewriter:
    def __init__(self, mapper: ConceptPrimitiveMapper, intent_classifier: IntentClassifier) -> None:
        self.mapper = mapper
        self.intent_classifier = intent_classifier

    def rewrite(
        self,
        question: str,
        detected_terms: Sequence[str],
        primitive_ids: Sequence[str],
        target_year: int,
    ) -> Tuple[str, str, float]:
        intent, confidence = self.intent_classifier.predict(question)
        phrases = self.mapper.primitive_phrases(primitive_ids)

        neutral_question = question
        sorted_terms = sorted(detected_terms, key=len, reverse=True)
        first_replacement_done = False
        for term in sorted_terms:
            if not first_replacement_done:
                replacement = "this method"
                first_replacement_done = True
            else:
                replacement = "this technical idea"
            neutral_question = replace_term(neutral_question, term, replacement)

        neutral_question = normalize_space(neutral_question)
        if not neutral_question.endswith("?"):
            neutral_question += "?"

        if phrases:
            if len(phrases) == 1:
                intro = f"Consider a method that {phrases[0]}."
            else:
                joined = "; and ".join(f"it {phrase}" for phrase in phrases)
                intro = f"Consider a technical arrangement: {joined}."
        else:
            intro = f"Restate the question using terms understandable before {target_year}."

        if intent == "process_explanation":
            suffix = "Focus on the mechanism, not on the modern name."
        elif intent == "purpose_explanation":
            suffix = "Focus on the practical purpose, not on the modern name."
        else:
            suffix = "Focus on the causal explanation, not on the modern name."

        rewritten = f"{intro} {neutral_question} {suffix}"
        return normalize_space(rewritten), intent, confidence


def ngram_set(tokens: Sequence[str], n: int) -> set[Tuple[str, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


class RuleBasedValidator:
    def __init__(self, primitive_dictionary: Dict[str, Any], config: PipelineConfig) -> None:
        self.primitive_dictionary = primitive_dictionary
        self.config = config

    def _primitive_recall(self, rewritten_question: str, required_primitives: Sequence[str]) -> Tuple[float, Dict[str, float]]:
        if not required_primitives:
            return 1.0, {}
        rewritten_tokens = set(tokenize(rewritten_question))
        per_primitive: Dict[str, float] = {}
        passed = 0
        for pid in required_primitives:
            primitive = self.primitive_dictionary.get(pid, {})
            keywords = primitive.get("keywords") or keyword_set(primitive.get("primitive_phrase", ""))
            if not keywords:
                recall = 1.0
            else:
                hits = sum(1 for word in keywords if word in rewritten_tokens)
                recall = hits / len(keywords)
            per_primitive[pid] = recall
            if recall >= self.config.min_primitive_keyword_recall:
                passed += 1
        return passed / len(required_primitives), per_primitive

    def _choice_copying_score(self, rewritten_question: str, choices: Dict[str, str]) -> float:
        q_tokens = tokenize(rewritten_question)
        q_ngrams = ngram_set(q_tokens, 4) | ngram_set(q_tokens, 5)
        max_score = 0.0
        for text in choices.values():
            c_tokens = tokenize(text)
            c_ngrams = ngram_set(c_tokens, 4) | ngram_set(c_tokens, 5)
            if q_ngrams and c_ngrams:
                score = len(q_ngrams & c_ngrams) / max(1, len(c_ngrams))
            else:
                common = set(q_tokens) & set(c_tokens)
                score = len(common) / max(1, len(set(c_tokens)))
            max_score = max(max_score, score)
        return max_score

    def validate(
        self,
        original_question: str,
        rewritten_question: str,
        choices: Dict[str, str],
        forbidden_terms: Sequence[str],
        required_primitives: Sequence[str],
    ) -> Dict[str, Any]:
        fail_reasons: List[str] = []
        forbidden_remaining = [term for term in forbidden_terms if contains_term(rewritten_question, term)]
        forbidden_term_count = len(forbidden_remaining)
        if forbidden_term_count:
            fail_reasons.append("forbidden_terms_remaining")

        primitive_recall, primitive_detail = self._primitive_recall(rewritten_question, required_primitives)
        if primitive_recall < 1.0:
            fail_reasons.append("required_primitives_missing")

        copying_score = self._choice_copying_score(rewritten_question, choices)
        choice_copying_risk = copying_score > self.config.max_choice_copying_score
        if choice_copying_risk:
            fail_reasons.append("choice_copying_risk")

        format_valid = not re.search(r"(^|\n)\s*[ABCD]\s*[\).:-]", rewritten_question) and "Answer:" not in rewritten_question
        if not format_valid:
            fail_reasons.append("format_invalid")

        length_ratio = len(rewritten_question) / max(1, len(original_question))
        length_valid = self.config.min_length_ratio <= length_ratio <= self.config.max_length_ratio
        if not length_valid:
            fail_reasons.append("length_out_of_range")

        leakage_patterns = [
            r"\bcorrect answer\b",
            r"\boption\s+[ABCD]\b",
            r"\bchoice\s+[ABCD]\b",
            r"\bfirst option\b",
            r"\bsecond option\b",
            r"\bthird option\b",
            r"\bfourth option\b",
        ]
        leakage_risk = any(re.search(pattern, rewritten_question, flags=re.IGNORECASE) for pattern in leakage_patterns)
        if leakage_risk:
            fail_reasons.append("leakage_risk")

        return {
            "forbidden_term_count": forbidden_term_count,
            "forbidden_terms_remaining": forbidden_remaining,
            "required_primitive_recall": primitive_recall,
            "required_primitive_detail": primitive_detail,
            "choice_copying_score": copying_score,
            "choice_copying_risk": choice_copying_risk,
            "format_valid": format_valid,
            "length_ratio": length_ratio,
            "length_valid": length_valid,
            "leakage_risk": leakage_risk,
            "pass": not fail_reasons,
            "fail_reasons": fail_reasons,
        }


class SelfRepairer:
    def __init__(self, mapper: ConceptPrimitiveMapper, primitive_dictionary: Dict[str, Any]) -> None:
        self.mapper = mapper
        self.primitive_dictionary = primitive_dictionary

    def repair(
        self,
        original_question: str,
        rewritten_question: str,
        validation_report: Dict[str, Any],
        forbidden_terms: Sequence[str],
        required_primitives: Sequence[str],
        target_year: int,
    ) -> str:
        revised = rewritten_question
        for term in forbidden_terms:
            revised = replace_term(revised, term, "this method")

        if "required_primitives_missing" in validation_report.get("fail_reasons", []):
            missing_phrases = []
            detail = validation_report.get("required_primitive_detail", {})
            for pid in required_primitives:
                if detail.get(pid, 0.0) < 1.0:
                    primitive = self.primitive_dictionary.get(pid)
                    if primitive:
                        missing_phrases.append(primitive["primitive_phrase"])
            if missing_phrases:
                repaired_intro = "Consider a method that " + "; and ".join(missing_phrases) + "."
                if revised.startswith("Consider a method that "):
                    revised = re.sub(
                        r"^Consider a method that .*?\.\s*",
                        repaired_intro + " ",
                        revised,
                        count=1,
                    )
                elif revised.startswith("Consider a technical arrangement:"):
                    revised = re.sub(
                        r"^Consider a technical arrangement: .*?\.\s*",
                        repaired_intro + " ",
                        revised,
                        count=1,
                    )
                else:
                    revised = repaired_intro + " " + revised

        if "format_invalid" in validation_report.get("fail_reasons", []):
            revised = re.sub(r"(^|\n)\s*[ABCD]\s*[\).:-].*", " ", revised)
            revised = revised.replace("Answer:", "")

        if "choice_copying_risk" in validation_report.get("fail_reasons", []):
            revised = revised.replace("It ", "The method ")

        return normalize_space(revised)


class EraNeutralPromptGenerator:
    def __init__(
        self,
        detector: AnachronismDetector,
        primitive_mapper: ConceptPrimitiveMapper,
        rewriter: EraNeutralRewriter,
        validator: RuleBasedValidator,
        repairer: SelfRepairer,
        config: PipelineConfig,
    ) -> None:
        self.detector = detector
        self.primitive_mapper = primitive_mapper
        self.rewriter = rewriter
        self.validator = validator
        self.repairer = repairer
        self.config = config

    def generate(self, item: Dict[str, Any]) -> RewriteResult:
        detected_terms, _sources = self.detector.detect_with_sources(item["original_question"])
        primitives = self.primitive_mapper.map_terms(detected_terms, item["original_question"])
        rewritten, intent_label, intent_confidence = self.rewriter.rewrite(
            question=item["original_question"],
            detected_terms=detected_terms,
            primitive_ids=primitives,
            target_year=item.get("target_year", self.config.target_year),
        )
        autoencoder_primitives = getattr(self.rewriter, "last_decoded_primitives", [])
        if autoencoder_primitives:
            primitives = autoencoder_primitives
        latent_summary = getattr(self.rewriter, "last_latent_summary", {})
        hparams = getattr(getattr(self.rewriter, "autoencoder", None), "hparams", None)
        hparams_dict = asdict(hparams) if hparams is not None else {}

        report = self.validator.validate(
            original_question=item["original_question"],
            rewritten_question=rewritten,
            choices=item["choices"],
            forbidden_terms=item.get("forbidden_terms", []),
            required_primitives=item.get("required_primitives", []),
        )

        n_repair = 0
        while not report["pass"] and n_repair < self.config.max_repair_attempts:
            rewritten = self.repairer.repair(
                original_question=item["original_question"],
                rewritten_question=rewritten,
                validation_report=report,
                forbidden_terms=item.get("forbidden_terms", []),
                required_primitives=item.get("required_primitives", []),
                target_year=item.get("target_year", self.config.target_year),
            )
            report = self.validator.validate(
                original_question=item["original_question"],
                rewritten_question=rewritten,
                choices=item["choices"],
                forbidden_terms=item.get("forbidden_terms", []),
                required_primitives=item.get("required_primitives", []),
            )
            n_repair += 1

        return RewriteResult(
            item_id=item["id"],
            original_question=item["original_question"],
            rewritten_question=rewritten,
            detected_terms=detected_terms,
            mapped_primitives=primitives,
            validation_report=report,
            n_repair_attempts=n_repair,
            pass_validation=report["pass"],
            intent_label=intent_label,
            intent_confidence=intent_confidence,
            autoencoder_hyperparams=hparams_dict,
            autoencoder_decoded_primitives=list(autoencoder_primitives),
            autoencoder_latent_summary=dict(latent_summary),
        )


def build_mcq_prompt(question: str, choices: Dict[str, str]) -> str:
    return textwrap.dedent(
        f"""
        You must answer the following multiple-choice question.
        Choose exactly one option among A, B, C, and D.
        Return only the letter.

        Question:
        {question}

        Choices:
        A. {choices["A"]}
        B. {choices["B"]}
        C. {choices["C"]}
        D. {choices["D"]}

        Answer:
        """
    ).strip()


def normalize_choice(raw_text: str) -> str:
    text = (raw_text or "").strip().upper()
    if re.fullmatch(r"[ABCD]", text):
        return text
    match = re.search(r"\b([ABCD])\b", text)
    if match:
        return match.group(1)
    match = re.search(r"OPTION\s*([ABCD])", text)
    if match:
        return match.group(1)
    match = re.search(r"CHOICE\s*([ABCD])", text)
    if match:
        return match.group(1)
    return INVALID


def normalize_choice_with_choices(raw_text: str, choices: Dict[str, str]) -> str:
    direct = normalize_choice(raw_text)
    if direct != INVALID:
        return direct

    raw_tokens = set(tokenize(raw_text))
    if not raw_tokens:
        return INVALID

    scored: List[Tuple[float, str]] = []
    for label, choice_text in choices.items():
        choice_tokens = set(tokenize(choice_text))
        if not choice_tokens:
            continue
        overlap = len(raw_tokens & choice_tokens) / len(choice_tokens)
        scored.append((overlap, label))

    if not scored:
        return INVALID
    scored.sort(reverse=True)
    best_score, best_label = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else 0.0
    if best_score >= 0.42 and best_score - second_score >= 0.08:
        return best_label
    return INVALID


class JsonlCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: Dict[str, Dict[str, Any]] = {}
        for row in read_jsonl(path):
            key = row.get("cache_key")
            if key:
                self.data[key] = row

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return self.data.get(key)

    def put(self, row: Dict[str, Any]) -> None:
        key = row.get("cache_key")
        if not key:
            return
        self.data[key] = row
        ensure_dir(self.path.parent)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


class MockTalkieClient:
    """
    Deterministic downstream simulator.

    This is useful for checking the full pipeline without relying on the Talkie
    web UI.  Metrics produced by this client must be reported as simulated, not
    as real Talkie evidence.
    """

    def __init__(self, cache_path: Path, seed: int = 13) -> None:
        self.cache = JsonlCache(cache_path)
        self.seed = seed

    def ask(
        self,
        prompt: str,
        item_id: str,
        condition: str,
        item: Dict[str, Any],
        rewrite_result: Optional[RewriteResult] = None,
    ) -> Dict[str, Any]:
        key = hashlib.sha256((condition + "\n" + prompt).encode("utf-8")).hexdigest()
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        gold = item["gold_answer"]
        validation_bonus = 0.08 if (rewrite_result and rewrite_result.pass_validation) else -0.04
        if condition == "proposed":
            p_correct = 0.62 + validation_bonus
            p_invalid = 0.025
        else:
            n_modern = len(item.get("gold_anachronism_terms", []))
            p_correct = max(0.22, 0.48 - 0.035 * n_modern)
            p_invalid = 0.075

        roll = (stable_hash(f"{self.seed}:{item_id}:{condition}") % 10_000) / 10_000
        if roll < p_correct:
            raw_response = gold
        elif roll < p_correct + p_invalid:
            raw_response = "I cannot select only one letter."
        else:
            raw_response = choose_wrong_label(gold, f"{item_id}:{condition}:wrong")

        row = {
            "item_id": item_id,
            "condition": condition,
            "prompt": prompt,
            "raw_response": raw_response,
            "cache_key": key,
            "provider": "mock_simulated_talkie",
        }
        self.cache.put(row)
        return row


class PlaywrightTalkieClient:
    """
    Experimental browser automation wrapper.

    Web UIs often change.  If this fails, use --mode prepare_manual and paste
    responses into the generated CSV instead.
    """

    def __init__(self, url: str, headless: bool, wait_timeout_ms: int, cache_path: Path) -> None:
        self.url = url
        self.headless = headless
        self.wait_timeout_ms = wait_timeout_ms
        self.cache = JsonlCache(cache_path)
        self._playwright = None
        self._browser = None
        self._page = None

    def __enter__(self) -> "PlaywrightTalkieClient":
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise RuntimeError("Playwright is not installed. Use --mode prepare_manual or install playwright.") from exc

        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=self.headless)
        self._page = self._browser.new_page()
        self._page.goto(self.url, wait_until="domcontentloaded", timeout=self.wait_timeout_ms)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._browser:
            self._browser.close()
        if self._playwright:
            self._playwright.stop()

    @staticmethod
    def _extract_talkie_response(body_text: str) -> str:
        if "TALKIE-1930" not in body_text:
            lines = [line.strip() for line in body_text.splitlines() if line.strip()]
            return lines[-1] if lines else ""
        tail = body_text.split("TALKIE-1930")[-1]
        cleaned_lines = []
        for line in tail.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped in {"New Chat", "Send"}:
                break
            if stripped == "Question:":
                break
            cleaned_lines.append(stripped)
        return normalize_space(" ".join(cleaned_lines))

    def _send_prompt_to_web_ui(self, prompt: str) -> str:
        assert self._page is not None
        page = self._page
        before_text = page.locator("body").inner_text(timeout=5000)
        before_count = before_text.count("TALKIE-1930")

        input_selectors = [
            "textarea.chat-textarea",
            "textarea[placeholder*='talkie' i]",
            "textarea",
            "[contenteditable='true']",
            "input[type='text']",
        ]
        input_box = None
        for selector in input_selectors:
            try:
                locator = page.locator(selector)
                count = locator.count()
                for idx in range(count):
                    candidate = locator.nth(idx)
                    if candidate.is_visible(timeout=1000):
                        candidate.wait_for(state="visible", timeout=5000)
                        input_box = candidate
                        break
                if input_box is not None:
                    break
            except Exception:
                continue
        if input_box is None:
            raise RuntimeError("Could not find Talkie input box.")

        try:
            input_box.fill(prompt)
        except Exception:
            input_box.click()
            page.keyboard.insert_text(prompt)

        sent = False
        send_selectors = [
            "button[type='submit']",
            "button:has-text('Send')",
            "button:has-text('Submit')",
            "[aria-label*='send' i]",
        ]
        for selector in send_selectors:
            try:
                buttons = page.locator(selector)
                for idx in range(buttons.count()):
                    button = buttons.nth(idx)
                    if button.is_visible(timeout=1000) and button.is_enabled(timeout=1000):
                        button.click(timeout=4000)
                        sent = True
                        break
                if sent:
                    break
            except Exception:
                continue
        if not sent:
            input_box.press("Enter")

        deadline = time.time() + (self.wait_timeout_ms / 1000.0)
        body_text = ""
        while time.time() < deadline:
            page.wait_for_timeout(500)
            body_text = page.locator("body").inner_text(timeout=5000)
            if body_text.count("TALKIE-1930") > before_count:
                page.wait_for_timeout(1800)
                break
        body_text = page.locator("body").inner_text(timeout=5000)
        return self._extract_talkie_response(body_text)

    def ask(
        self,
        prompt: str,
        item_id: str,
        condition: str,
        item: Optional[Dict[str, Any]] = None,
        rewrite_result: Optional[RewriteResult] = None,
    ) -> Dict[str, Any]:
        key = hashlib.sha256((condition + "\n" + prompt).encode("utf-8")).hexdigest()
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        raw_response = self._send_prompt_to_web_ui(prompt)
        row = {
            "item_id": item_id,
            "condition": condition,
            "prompt": prompt,
            "raw_response": raw_response,
            "cache_key": key,
            "provider": "talkie_web_playwright",
        }
        self.cache.put(row)
        return row


def term_eval_counts(predicted: Sequence[str], gold: Sequence[str]) -> Tuple[int, int, int]:
    pred_set = {normalize_term(x) for x in predicted}
    gold_set = {normalize_term(x) for x in gold}
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    return tp, fp, fn


def build_result_row(
    item: Dict[str, Any],
    rewrite_result: RewriteResult,
    baseline_prompt: str,
    proposed_prompt: str,
    baseline_response: Dict[str, Any],
    proposed_response: Dict[str, Any],
) -> Dict[str, Any]:
    baseline_pred = normalize_choice_with_choices(baseline_response.get("raw_response", ""), item["choices"])
    proposed_pred = normalize_choice_with_choices(proposed_response.get("raw_response", ""), item["choices"])
    gold = item["gold_answer"]
    detection_tp, detection_fp, detection_fn = term_eval_counts(
        rewrite_result.detected_terms,
        item.get("gold_anachronism_terms", []),
    )
    report = rewrite_result.validation_report
    return {
        "id": item["id"],
        "domain": item["domain"],
        "gold_answer": gold,
        "original_question": item["original_question"],
        "rewritten_question": rewrite_result.rewritten_question,
        "baseline_prompt": baseline_prompt,
        "proposed_prompt": proposed_prompt,
        "baseline_raw_response": baseline_response.get("raw_response", ""),
        "proposed_raw_response": proposed_response.get("raw_response", ""),
        "baseline_pred": baseline_pred,
        "proposed_pred": proposed_pred,
        "baseline_correct": baseline_pred == gold,
        "proposed_correct": proposed_pred == gold,
        "detected_terms": json_dumps(rewrite_result.detected_terms),
        "mapped_primitives": json_dumps(rewrite_result.mapped_primitives),
        "autoencoder_hyperparams": json_dumps(rewrite_result.autoencoder_hyperparams),
        "autoencoder_decoded_primitives": json_dumps(rewrite_result.autoencoder_decoded_primitives),
        "autoencoder_latent_summary": json_dumps(rewrite_result.autoencoder_latent_summary),
        "detection_tp": detection_tp,
        "detection_fp": detection_fp,
        "detection_fn": detection_fn,
        "forbidden_term_count": report.get("forbidden_term_count", 0),
        "required_primitive_recall": report.get("required_primitive_recall", 0.0),
        "choice_copying_score": report.get("choice_copying_score", 0.0),
        "leakage_risk": report.get("leakage_risk", False),
        "rewrite_validation_pass": rewrite_result.pass_validation,
        "n_repair_attempts": rewrite_result.n_repair_attempts,
        "intent_label": rewrite_result.intent_label,
        "intent_confidence": rewrite_result.intent_confidence,
        "validation_fail_reasons": json_dumps(report.get("fail_reasons", [])),
    }


def compute_class_metrics(rows: Sequence[Dict[str, Any]], pred_key: str) -> Dict[str, Any]:
    n = len(rows)
    correct = sum(1 for row in rows if row[pred_key] == row["gold_answer"])
    invalid = sum(1 for row in rows if row[pred_key] == INVALID)
    per_label = {}
    f1s = []
    for label in LABELS:
        tp = sum(1 for row in rows if row[pred_key] == label and row["gold_answer"] == label)
        fp = sum(1 for row in rows if row[pred_key] == label and row["gold_answer"] != label)
        fn = sum(1 for row in rows if row[pred_key] != label and row["gold_answer"] == label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
        f1s.append(f1)
    return {
        "accuracy": correct / n if n else 0.0,
        "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "invalid_rate": invalid / n if n else 0.0,
        "n_items": n,
        "per_label": per_label,
    }


def compute_downstream_metrics(rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    baseline = compute_class_metrics(rows, "baseline_pred")
    proposed = compute_class_metrics(rows, "proposed_pred")
    metrics_rows = [
        {
            "method": "baseline_raw_prompt",
            "accuracy": baseline["accuracy"],
            "macro_f1": baseline["macro_f1"],
            "invalid_rate": baseline["invalid_rate"],
            "n_items": baseline["n_items"],
        },
        {
            "method": "proposed_era_neutral_prompt",
            "accuracy": proposed["accuracy"],
            "macro_f1": proposed["macro_f1"],
            "invalid_rate": proposed["invalid_rate"],
            "n_items": proposed["n_items"],
        },
        {
            "method": "improvement",
            "accuracy": proposed["accuracy"] - baseline["accuracy"],
            "macro_f1": proposed["macro_f1"] - baseline["macro_f1"],
            "invalid_rate": proposed["invalid_rate"] - baseline["invalid_rate"],
            "n_items": baseline["n_items"],
        },
    ]
    detail = {"baseline": baseline, "proposed": proposed}
    return metrics_rows, detail


def compute_component_metrics(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    tp = sum(int(row["detection_tp"]) for row in rows)
    fp = sum(int(row["detection_fp"]) for row in rows)
    fn = sum(int(row["detection_fn"]) for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    n = len(rows) or 1
    removal_rate = sum(1 for row in rows if int(row["forbidden_term_count"]) == 0) / n
    primitive_recall = sum(float(row["required_primitive_recall"]) for row in rows) / n
    pass_rate = sum(1 for row in rows if row["rewrite_validation_pass"] in (True, "True", "true", 1, "1")) / n
    leakage_rate = sum(1 for row in rows if row["leakage_risk"] in (True, "True", "true", 1, "1")) / n
    mean_repair = sum(int(row["n_repair_attempts"]) for row in rows) / n

    return [
        {"component": "detector", "metric": "precision", "value": precision},
        {"component": "detector", "metric": "recall", "value": recall},
        {"component": "detector", "metric": "f1", "value": f1},
        {"component": "rewriter", "metric": "anachronism_removal_rate", "value": removal_rate},
        {"component": "rewriter", "metric": "required_primitive_recall", "value": primitive_recall},
        {"component": "validator", "metric": "rewrite_pass_rate", "value": pass_rate},
        {"component": "validator", "metric": "leakage_risk_rate", "value": leakage_rate},
        {"component": "repair", "metric": "mean_repair_attempts", "value": mean_repair},
    ]


def compute_mcnemar(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    b = 0  # baseline wrong, proposed right
    c = 0  # baseline right, proposed wrong
    for row in rows:
        base = row["baseline_correct"] in (True, "True", "true", 1, "1")
        prop = row["proposed_correct"] in (True, "True", "true", 1, "1")
        if not base and prop:
            b += 1
        elif base and not prop:
            c += 1
    if b + c == 0:
        chi2 = 0.0
        p_value = 1.0
    else:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = math.erfc(math.sqrt(max(chi2, 0.0) / 2.0))
    return {"baseline_wrong_proposed_right": b, "baseline_right_proposed_wrong": c, "chi2": chi2, "p_value": p_value}


def confusion_matrix(rows: Sequence[Dict[str, Any]], pred_key: str) -> List[List[int]]:
    labels = LABELS + [INVALID]
    idx = {label: i for i, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for row in rows:
        gold = row["gold_answer"]
        pred = row[pred_key] if row[pred_key] in idx else INVALID
        matrix[idx[gold]][idx[pred]] += 1
    return matrix


def _write_png(path: Path, width: int, height: int, pixels: List[List[Tuple[int, int, int]]]) -> None:
    """Write an RGB PNG using only the standard library."""
    raw = bytearray()
    for y in range(height):
        raw.append(0)
        for x in range(width):
            raw.extend(bytes(pixels[y][x]))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(bytes(raw), level=9))
    png += chunk(b"IEND", b"")
    path.write_bytes(png)


def _canvas(width: int, height: int, color: Tuple[int, int, int] = (255, 255, 255)) -> List[List[Tuple[int, int, int]]]:
    return [[color for _ in range(width)] for _ in range(height)]


def _rect(
    pixels: List[List[Tuple[int, int, int]]],
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: Tuple[int, int, int],
) -> None:
    height = len(pixels)
    width = len(pixels[0]) if height else 0
    x0 = max(0, min(width, x0))
    x1 = max(0, min(width, x1))
    y0 = max(0, min(height, y0))
    y1 = max(0, min(height, y1))
    for y in range(y0, y1):
        row = pixels[y]
        for x in range(x0, x1):
            row[x] = color


def _line_h(pixels: List[List[Tuple[int, int, int]]], x0: int, x1: int, y: int, color: Tuple[int, int, int]) -> None:
    _rect(pixels, x0, y, x1, y + 1, color)


def _line_v(pixels: List[List[Tuple[int, int, int]]], x: int, y0: int, y1: int, color: Tuple[int, int, int]) -> None:
    _rect(pixels, x, y0, x + 1, y1, color)


def _bar_png(path: Path, values: Sequence[float], colors: Sequence[Tuple[int, int, int]]) -> None:
    width, height = 640, 420
    pixels = _canvas(width, height)
    left, right, top, bottom = 70, 40, 35, 55
    chart_w = width - left - right
    chart_h = height - top - bottom
    axis = (40, 40, 40)
    grid = (226, 232, 240)
    _line_v(pixels, left, top, top + chart_h, axis)
    _line_h(pixels, left, left + chart_w, top + chart_h, axis)
    for i in range(1, 5):
        y = top + chart_h - int(chart_h * i / 5)
        _line_h(pixels, left, left + chart_w, y, grid)
    n = max(1, len(values))
    slot = chart_w // n
    bar_w = int(slot * 0.48)
    for i, value in enumerate(values):
        value = max(0.0, min(1.0, float(value)))
        x_mid = left + slot * i + slot // 2
        x0 = x_mid - bar_w // 2
        x1 = x_mid + bar_w // 2
        y1 = top + chart_h
        y0 = y1 - int(chart_h * value)
        _rect(pixels, x0, y0, x1, y1, colors[i % len(colors)])
    _write_png(path, width, height, pixels)


def _heatmap_png(path: Path, matrix: List[List[int]]) -> None:
    cell = 72
    margin = 42
    size = margin * 2 + cell * len(matrix)
    pixels = _canvas(size, size)
    max_value = max([max(row) for row in matrix] or [1]) or 1
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            shade = int(245 - 170 * (value / max_value))
            color = (shade, shade + 5 if shade < 245 else 245, 255)
            x0 = margin + j * cell
            y0 = margin + i * cell
            _rect(pixels, x0, y0, x0 + cell - 2, y0 + cell - 2, color)
    grid_color = (80, 80, 80)
    for k in range(len(matrix) + 1):
        pos = margin + k * cell
        _line_v(pixels, pos, margin, margin + cell * len(matrix), grid_color)
        _line_h(pixels, margin, margin + cell * len(matrix), pos, grid_color)
    _write_png(path, size, size, pixels)


def save_figures(rows: Sequence[Dict[str, Any]], final_metrics: Sequence[Dict[str, Any]], component_metrics: Sequence[Dict[str, Any]], out_dir: Path) -> None:
    figures_dir = out_dir / "figures"
    ensure_dir(figures_dir)

    metric_by_method = {row["method"]: row for row in final_metrics}
    methods = ["baseline_raw_prompt", "proposed_era_neutral_prompt"]

    _bar_png(
        figures_dir / "accuracy_comparison.png",
        [metric_by_method[m]["accuracy"] for m in methods],
        [(107, 114, 128), (37, 99, 235)],
    )
    _bar_png(
        figures_dir / "macro_f1_comparison.png",
        [metric_by_method[m]["macro_f1"] for m in methods],
        [(107, 114, 128), (37, 99, 235)],
    )

    _heatmap_png(figures_dir / "confusion_baseline.png", confusion_matrix(rows, "baseline_pred"))
    _heatmap_png(figures_dir / "confusion_proposed.png", confusion_matrix(rows, "proposed_pred"))

    pred_counts = {
        "Baseline": Counter(row["baseline_pred"] for row in rows),
        "Proposed": Counter(row["proposed_pred"] for row in rows),
    }
    labels5 = LABELS + [INVALID]
    max_count = max([pred_counts[name].get(label, 0) for name in pred_counts for label in labels5] or [1])
    distribution_values: List[float] = []
    distribution_colors: List[Tuple[int, int, int]] = []
    for label in labels5:
        distribution_values.append(pred_counts["Baseline"].get(label, 0) / max_count)
        distribution_colors.append((107, 114, 128))
        distribution_values.append(pred_counts["Proposed"].get(label, 0) / max_count)
        distribution_colors.append((37, 99, 235))
    _bar_png(figures_dir / "label_distribution.png", distribution_values, distribution_colors)

    component_values = {
        row["metric"]: row["value"]
        for row in component_metrics
        if row["metric"] in {"f1", "anachronism_removal_rate", "required_primitive_recall", "rewrite_pass_rate"}
    }
    comp_metrics = ["f1", "anachronism_removal_rate", "required_primitive_recall", "rewrite_pass_rate"]
    _bar_png(
        figures_dir / "generator_component_metrics.png",
        [component_values.get(metric, 0.0) for metric in comp_metrics],
        [(5, 150, 105), (16, 185, 129), (20, 184, 166), (13, 148, 136)],
    )


def pick_qualitative_examples(rows: Sequence[Dict[str, Any]], max_each: int = 3) -> Dict[str, List[Dict[str, Any]]]:
    success = [r for r in rows if not r["baseline_correct"] and r["proposed_correct"]]
    regression = [r for r in rows if r["baseline_correct"] and not r["proposed_correct"]]
    invalid_fixed = [r for r in rows if r["baseline_pred"] == INVALID and r["proposed_pred"] != INVALID]
    validation_fail = [r for r in rows if not r["rewrite_validation_pass"]]
    return {
        "success_cases": success[:max_each],
        "regression_cases": regression[:max_each],
        "invalid_fixed_cases": invalid_fixed[:max_each],
        "validation_fail_cases": validation_fail[:max_each],
    }


def write_qualitative_examples(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    examples = pick_qualitative_examples(rows)
    lines = ["# Qualitative Examples", ""]
    for section, section_rows in examples.items():
        lines.append(f"## {section.replace('_', ' ').title()}")
        lines.append("")
        if not section_rows:
            lines.append("No examples in this category.")
            lines.append("")
            continue
        for row in section_rows:
            lines.append(f"### {row['id']} ({row['domain']})")
            lines.append("")
            lines.append(f"- Gold: `{row['gold_answer']}`")
            lines.append(f"- Baseline pred: `{row['baseline_pred']}`")
            lines.append(f"- Proposed pred: `{row['proposed_pred']}`")
            lines.append(f"- Original: {row['original_question']}")
            lines.append(f"- Rewritten: {row['rewritten_question']}")
            lines.append(f"- Detected terms: `{row['detected_terms']}`")
            lines.append(f"- Validation failures: `{row['validation_fail_reasons']}`")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def metric_lookup(rows: Sequence[Dict[str, Any]], method: str, metric: str) -> float:
    for row in rows:
        if row["method"] == method:
            return float(row[metric])
    return 0.0


def generate_report(
    rows: Sequence[Dict[str, Any]],
    final_metrics: Sequence[Dict[str, Any]],
    component_metrics: Sequence[Dict[str, Any]],
    mcnemar: Dict[str, Any],
    out_dir: Path,
    mode: str,
    provider_name: str,
    selected_hparams: Optional[AutoencoderHyperParams] = None,
    grid_rows: Optional[Sequence[Dict[str, Any]]] = None,
) -> None:
    baseline_acc = metric_lookup(final_metrics, "baseline_raw_prompt", "accuracy")
    proposed_acc = metric_lookup(final_metrics, "proposed_era_neutral_prompt", "accuracy")
    baseline_f1 = metric_lookup(final_metrics, "baseline_raw_prompt", "macro_f1")
    proposed_f1 = metric_lookup(final_metrics, "proposed_era_neutral_prompt", "macro_f1")
    improvement = proposed_acc - baseline_acc
    component = {(row["component"], row["metric"]): float(row["value"]) for row in component_metrics}

    if provider_name == "mock_simulated_talkie":
        evidence_note = (
            "Important: downstream responses in this run were produced by the deterministic "
            "mock evaluator. They are useful for pipeline verification, but real project "
            "claims should be based on manual or web Talkie responses."
        )
    elif provider_name == "talkie_web_playwright":
        evidence_note = (
            "Raw and era-neutral prompts were sent to https://talkie-lm.com/chat through "
            "the Playwright Talkie web client, and responses were scraped into the result logs."
        )
    else:
        evidence_note = "Downstream responses were collected from the configured Talkie evaluation provider."

    lines = [
        "# Term Project Team 2 Report",
        "",
        "## Project Definition",
        "",
        "The proposed model is the Era-Neutral Prompt Generator. Talkie 1930 is treated as a frozen downstream evaluator, not as the model being improved.",
        "",
        "The experiment uses a paired design: each question is evaluated twice with the same choices and same gold answer. The baseline condition uses the raw modern question. The proposed condition uses only the rewritten question produced by the prompt generator.",
        "",
        "## Run Metadata",
        "",
        f"- Mode: `{mode}`",
        f"- Downstream provider: `{provider_name}`",
        f"- Number of items: `{len(rows)}`",
        f"- Note: {evidence_note}",
        "",
        "## Proposed Model Algorithm",
        "",
        "The Era-Neutral Prompt Generator is implemented as a denoising text Autoencoder. The encoder maps a modern question bag-of-words into a compressed latent vector. The decoder reconstructs era-neutral primitive tokens from that latent representation. The highest-scoring decoded primitive is then used to build the final 1930-era-neutral question. The validator checks forbidden modern terms, primitive preservation, answer-choice copying, formatting, length, and leakage risk; failed rewrites are repaired and revalidated.",
        "",
        "Pipeline:",
        "",
        "```text",
        "modern question -> detector -> autoencoder encoder -> latent bottleneck",
        "                -> decoder -> primitive reconstruction -> era-neutral rewrite",
        "                -> validator/self-repair -> Talkie 1930 evaluation",
        "```",
        "",
        "## Autoencoder Hyperparameter Pool",
        "",
        "| Candidate | Latent Dim | LR | Epochs | Noise | L2 | Decode Threshold | Top-K |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, hp in enumerate(default_autoencoder_hyperparameter_pool(), start=1):
        lines.append(
            f"| {idx} | {hp.latent_dim} | {hp.learning_rate:.4f} | {hp.epochs} | "
            f"{hp.noise_prob:.2f} | {hp.l2:.4f} | {hp.decode_threshold:.3f} | {hp.candidate_top_k} |"
        )

    lines += [
        "",
        "Grid search selection metric is Talkie downstream performance on the configured grid-evaluation subset: highest Talkie accuracy, then highest Macro-F1, then lowest invalid rate.",
        "",
    ]
    if selected_hparams is not None:
        lines += [
            "## Selected Autoencoder Hyperparameters",
            "",
            f"- Candidate id: `{selected_hparams.id}`",
            f"- Latent dimension: `{selected_hparams.latent_dim}`",
            f"- Learning rate: `{selected_hparams.learning_rate}`",
            f"- Epochs: `{selected_hparams.epochs}`",
            f"- Denoising drop probability: `{selected_hparams.noise_prob}`",
            f"- L2 regularization: `{selected_hparams.l2}`",
            f"- Decode threshold: `{selected_hparams.decode_threshold}`",
            f"- Primitive top-k: `{selected_hparams.candidate_top_k}`",
            "",
        ]
    if grid_rows:
        lines += [
            "## Autoencoder Grid Search Results",
            "",
            "| Rank | Candidate | Talkie Acc. | Talkie Macro-F1 | Invalid Rate | Rewrite Pass |",
            "|---:|---|---:|---:|---:|---:|",
        ]
        for rank, row in enumerate(grid_rows, start=1):
            lines.append(
                f"| {rank} | `{row['candidate_id']}` | {float(row['talkie_accuracy']):.4f} | "
                f"{float(row['talkie_macro_f1']):.4f} | {float(row['talkie_invalid_rate']):.4f} | "
                f"{float(row['rewrite_pass_rate']):.4f} |"
            )
        lines.append("")

    lines += [
        "## Downstream Utility Metrics",
        "",
        "| Method | Accuracy | Macro-F1 | Invalid Rate | N |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in final_metrics:
        lines.append(
            f"| {row['method']} | {float(row['accuracy']):.4f} | {float(row['macro_f1']):.4f} | "
            f"{float(row['invalid_rate']):.4f} | {int(row['n_items'])} |"
        )
    lines += [
        "",
        f"Accuracy improvement: `{improvement:+.4f}`",
        f"Macro-F1 improvement: `{proposed_f1 - baseline_f1:+.4f}`",
        "",
        "## Paired Test",
        "",
        f"- Baseline wrong, proposed right: `{mcnemar['baseline_wrong_proposed_right']}`",
        f"- Baseline right, proposed wrong: `{mcnemar['baseline_right_proposed_wrong']}`",
        f"- McNemar chi-square approximation: `{mcnemar['chi2']:.4f}`",
        f"- Approximate p-value: `{mcnemar['p_value']:.4f}`",
        "",
        "## Generator Component Metrics",
        "",
        "| Component | Metric | Value |",
        "|---|---|---:|",
    ]
    for row in component_metrics:
        lines.append(f"| {row['component']} | {row['metric']} | {float(row['value']):.4f} |")

    examples = pick_qualitative_examples(rows, max_each=2)
    lines += [
        "",
        "## Interpretation",
        "",
        "If proposed accuracy is higher than baseline accuracy under a real Talkie provider, the correct interpretation is not that Talkie changed. The downstream model is fixed. The improvement is evidence that the proposed preprocessor reduced anachronistic vocabulary barriers in the input.",
        "",
        "The internal component metrics check whether the generator actually performed the intended transformation: detector F1 for modern-term discovery, anachronism removal rate for forbidden term removal, primitive recall for meaning preservation, and validator pass rate for experimental fairness.",
        "",
        "## Example Improvements",
        "",
    ]
    for row in examples["success_cases"]:
        lines += [
            f"### {row['id']}",
            "",
            f"- Original: {row['original_question']}",
            f"- Rewritten: {row['rewritten_question']}",
            f"- Gold / baseline / proposed: `{row['gold_answer']}` / `{row['baseline_pred']}` / `{row['proposed_pred']}`",
            "",
        ]
    if not examples["success_cases"]:
        lines.append("No baseline-wrong/proposed-right examples were found in this run.")
        lines.append("")

    lines += [
        "## Failure Modes To Discuss",
        "",
        "- A rewrite can pass term removal while still making the question too verbose.",
        "- Dictionary detection may miss unseen modern expressions unless the ML span classifier or dictionary is expanded.",
        "- Choice-copying validation is conservative and should be manually reviewed for final reporting.",
        "- Web automation can be brittle; cached or manually pasted Talkie outputs are safer for final reproducibility.",
        "",
        "## Generated Artifacts",
        "",
        "- `input_data/raw_4choice_questions.csv` / `.jsonl`",
        "- `input_data/era_neutral_preprocessed_questions.csv` / `.jsonl`",
        "- `per_item_results.csv` / `per_item_results.jsonl`",
        "- `final_metrics.csv`",
        "- `component_metrics.csv`",
        "- `selected_autoencoder_hyperparameters.json`",
        "- `autoencoder_grid_search_results.csv` when grid-search mode is used",
        "- `qualitative_examples.md`",
        "- `figures/`",
    ]

    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def prepare_manual_sheet(prepared_rows: Sequence[Dict[str, Any]], out_dir: Path) -> None:
    manual_rows: List[Dict[str, Any]] = []
    for row in prepared_rows:
        manual_rows.append(
            {
                "item_id": row["id"],
                "condition": "baseline",
                "prompt": row["baseline_prompt"],
                "raw_response_manual": "",
            }
        )
        manual_rows.append(
            {
                "item_id": row["id"],
                "condition": "proposed",
                "prompt": row["proposed_prompt"],
                "raw_response_manual": "",
            }
        )
    write_csv_dicts(out_dir / "manual_talkie_input_sheet.csv", manual_rows)


def load_manual_responses(path: Path) -> Dict[Tuple[str, str], str]:
    rows = read_csv_dicts(path)
    responses: Dict[Tuple[str, str], str] = {}
    for row in rows:
        item_id = row.get("item_id") or row.get("id")
        condition = row.get("condition")
        raw = row.get("raw_response_manual") or row.get("raw_response") or ""
        if item_id and condition:
            responses[(item_id, condition)] = raw
    return responses


def write_input_data_artifacts(input_dir: Path, dataset: Sequence[Dict[str, Any]], rows: Sequence[Dict[str, Any]]) -> None:
    ensure_dir(input_dir)
    raw_rows = []
    for item in dataset:
        raw_rows.append(
            {
                "id": item["id"],
                "domain": item["domain"],
                "target_year": item["target_year"],
                "original_question": item["original_question"],
                "choice_A": item["choices"]["A"],
                "choice_B": item["choices"]["B"],
                "choice_C": item["choices"]["C"],
                "choice_D": item["choices"]["D"],
                "gold_answer": item["gold_answer"],
                "gold_anachronism_terms": json_dumps(item.get("gold_anachronism_terms", [])),
                "required_primitives": json_dumps(item.get("required_primitives", [])),
            }
        )

    preprocessed_rows = []
    for row in rows:
        preprocessed_rows.append(
            {
                "id": row["id"],
                "domain": row["domain"],
                "original_question": row["original_question"],
                "rewritten_question": row["rewritten_question"],
                "baseline_prompt": row["baseline_prompt"],
                "proposed_prompt": row["proposed_prompt"],
                "detected_terms": row["detected_terms"],
                "mapped_primitives": row["mapped_primitives"],
                "autoencoder_hyperparams": row.get("autoencoder_hyperparams", ""),
                "autoencoder_decoded_primitives": row.get("autoencoder_decoded_primitives", ""),
                "autoencoder_latent_summary": row.get("autoencoder_latent_summary", ""),
                "rewrite_validation_pass": row["rewrite_validation_pass"],
            }
        )

    write_csv_dicts(input_dir / "raw_4choice_questions.csv", raw_rows)
    write_jsonl(input_dir / "raw_4choice_questions.jsonl", raw_rows)
    write_csv_dicts(input_dir / "era_neutral_preprocessed_questions.csv", preprocessed_rows)
    write_jsonl(input_dir / "era_neutral_preprocessed_questions.jsonl", preprocessed_rows)


def build_generator(
    config: PipelineConfig,
    modern_dictionary: Dict[str, Any],
    primitive_dictionary: Dict[str, Any],
    dataset: Sequence[Dict[str, Any]],
    autoencoder_hparams: Optional[AutoencoderHyperParams] = None,
) -> EraNeutralPromptGenerator:
    detector = AnachronismDetector(modern_dictionary)
    mapper = ConceptPrimitiveMapper(primitive_dictionary, modern_dictionary)
    intent_classifier = IntentClassifier()
    selected_hparams = autoencoder_hparams or default_autoencoder_hyperparameter_pool()[0]
    autoencoder = DenoisingTextAutoencoder(selected_hparams, seed=config.seed)
    autoencoder.fit(build_autoencoder_training_pairs(dataset, primitive_dictionary))
    rewriter = AutoencoderEraNeutralRewriter(mapper, intent_classifier, autoencoder, primitive_dictionary)
    validator = RuleBasedValidator(primitive_dictionary, config)
    repairer = SelfRepairer(mapper, primitive_dictionary)
    return EraNeutralPromptGenerator(detector, mapper, rewriter, validator, repairer, config)


def run_autoencoder_grid_search(
    dataset: Sequence[Dict[str, Any]],
    modern_dictionary: Dict[str, Any],
    primitive_dictionary: Dict[str, Any],
    config: PipelineConfig,
    client: Any,
    out_dir: Path,
    grid_eval_items: int,
) -> Tuple[AutoencoderHyperParams, List[Dict[str, Any]]]:
    ensure_dir(out_dir)
    pool = default_autoencoder_hyperparameter_pool()
    eval_dataset = list(dataset[: max(1, min(grid_eval_items, len(dataset)))])
    grid_rows: List[Dict[str, Any]] = []

    for candidate_index, hparams in enumerate(pool, start=1):
        generator = build_generator(
            config,
            modern_dictionary,
            primitive_dictionary,
            dataset,
            autoencoder_hparams=hparams,
        )
        candidate_rows: List[Dict[str, Any]] = []
        for item in eval_dataset:
            rewrite_result = generator.generate(item)
            proposed_prompt = build_mcq_prompt(rewrite_result.rewritten_question, item["choices"])
            response = client.ask(
                proposed_prompt,
                item_id=item["id"],
                condition=f"grid_candidate_{candidate_index}_{hparams.id}",
                item=item,
                rewrite_result=rewrite_result,
            )
            pred = normalize_choice_with_choices(response.get("raw_response", ""), item["choices"])
            candidate_rows.append(
                {
                    "gold_answer": item["gold_answer"],
                    "proposed_pred": pred,
                    "rewrite_validation_pass": rewrite_result.pass_validation,
                    "forbidden_term_count": rewrite_result.validation_report.get("forbidden_term_count", 0),
                    "required_primitive_recall": rewrite_result.validation_report.get("required_primitive_recall", 0.0),
                    "leakage_risk": rewrite_result.validation_report.get("leakage_risk", False),
                    "n_repair_attempts": rewrite_result.n_repair_attempts,
                    "detection_tp": term_eval_counts(
                        rewrite_result.detected_terms,
                        item.get("gold_anachronism_terms", []),
                    )[0],
                    "detection_fp": term_eval_counts(
                        rewrite_result.detected_terms,
                        item.get("gold_anachronism_terms", []),
                    )[1],
                    "detection_fn": term_eval_counts(
                        rewrite_result.detected_terms,
                        item.get("gold_anachronism_terms", []),
                    )[2],
                }
            )

        metrics = compute_class_metrics(candidate_rows, "proposed_pred")
        pass_rate = (
            sum(1 for row in candidate_rows if row["rewrite_validation_pass"]) / len(candidate_rows)
            if candidate_rows
            else 0.0
        )
        grid_rows.append(
            {
                "candidate_index": candidate_index,
                "candidate_id": hparams.id,
                "latent_dim": hparams.latent_dim,
                "learning_rate": hparams.learning_rate,
                "epochs": hparams.epochs,
                "noise_prob": hparams.noise_prob,
                "l2": hparams.l2,
                "decode_threshold": hparams.decode_threshold,
                "candidate_top_k": hparams.candidate_top_k,
                "talkie_eval_items": len(eval_dataset),
                "talkie_accuracy": metrics["accuracy"],
                "talkie_macro_f1": metrics["macro_f1"],
                "talkie_invalid_rate": metrics["invalid_rate"],
                "rewrite_pass_rate": pass_rate,
            }
        )

    grid_rows = sorted(
        grid_rows,
        key=lambda row: (
            -float(row["talkie_accuracy"]),
            -float(row["talkie_macro_f1"]),
            float(row["talkie_invalid_rate"]),
            -float(row["rewrite_pass_rate"]),
            int(row["candidate_index"]),
        ),
    )
    write_csv_dicts(
        out_dir / "autoencoder_grid_search_results.csv",
        grid_rows,
        [
            "candidate_index",
            "candidate_id",
            "latent_dim",
            "learning_rate",
            "epochs",
            "noise_prob",
            "l2",
            "decode_threshold",
            "candidate_top_k",
            "talkie_eval_items",
            "talkie_accuracy",
            "talkie_macro_f1",
            "talkie_invalid_rate",
            "rewrite_pass_rate",
        ],
    )
    selected_row = grid_rows[0]
    selected = AutoencoderHyperParams(
        latent_dim=int(selected_row["latent_dim"]),
        learning_rate=float(selected_row["learning_rate"]),
        epochs=int(selected_row["epochs"]),
        noise_prob=float(selected_row["noise_prob"]),
        l2=float(selected_row["l2"]),
        decode_threshold=float(selected_row["decode_threshold"]),
        candidate_top_k=int(selected_row["candidate_top_k"]),
    )
    write_json(out_dir / "selected_autoencoder_hyperparameters.json", asdict(selected))
    return selected, grid_rows


def materialize_dataset_and_dicts(args: argparse.Namespace, config: PipelineConfig, data_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    ensure_dir(data_dir)
    concepts = build_concept_library()
    modern_dictionary, primitive_dictionary = build_dictionaries(concepts)

    data_path = data_dir / "generated_questions.jsonl"
    should_regenerate = args.force_regenerate_data or not data_path.exists()
    if should_regenerate:
        dataset = generate_dataset(config.n_items, config.target_year, config.seed, concepts)
        write_jsonl(data_path, dataset)
    else:
        dataset = read_jsonl(data_path)
        if len(dataset) < config.n_items:
            dataset = generate_dataset(config.n_items, config.target_year, config.seed, concepts)
            write_jsonl(data_path, dataset)
        else:
            dataset = dataset[: config.n_items]

    write_json(data_dir / "modern_terms_dictionary.json", modern_dictionary)
    write_json(data_dir / "primitive_dictionary.json", primitive_dictionary)
    return dataset, modern_dictionary, primitive_dictionary


def run_pipeline(args: argparse.Namespace) -> None:
    config = PipelineConfig(
        n_items=args.n_items,
        target_year=args.target_year,
        seed=args.seed,
        max_repair_attempts=args.max_repair_attempts,
        talkie_wait_timeout_ms=args.talkie_wait_timeout_ms,
    )

    project_dir = Path.cwd()
    data_dir = project_dir / "data"
    input_dir = project_dir / "input_data"
    cache_dir = project_dir / "cache"
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = project_dir / out_dir
    ensure_dir(cache_dir)
    ensure_dir(out_dir)

    dataset, modern_dictionary, primitive_dictionary = materialize_dataset_and_dicts(args, config, data_dir)

    manual_responses: Dict[Tuple[str, str], str] = {}
    if args.mode == "evaluate_manual":
        manual_path = Path(args.manual_response_csv)
        if not manual_path.is_absolute():
            manual_path = project_dir / manual_path
        if not manual_path.exists():
            raise FileNotFoundError(f"Manual response CSV not found: {manual_path}")
        manual_responses = load_manual_responses(manual_path)

    prepared_rows: List[Dict[str, Any]] = []
    result_rows: List[Dict[str, Any]] = []
    rewrite_cache_rows: List[Dict[str, Any]] = []
    selected_hparams = default_autoencoder_hyperparameter_pool()[0]
    grid_rows: List[Dict[str, Any]] = []

    client: Any = None
    client_context = None
    provider_name = "none"
    if args.mode in {"simulate", "simulate_grid"}:
        client = MockTalkieClient(cache_dir / "talkie_mock_cache.jsonl", seed=args.seed)
        provider_name = "mock_simulated_talkie"
    elif args.mode in {"run_web", "run_web_grid"}:
        client_context = PlaywrightTalkieClient(
            url=args.talkie_url,
            headless=args.headless,
            wait_timeout_ms=args.talkie_wait_timeout_ms,
            cache_path=cache_dir / "talkie_web_cache.jsonl",
        )
        client = client_context.__enter__()
        provider_name = "talkie_web_playwright"
    elif args.mode == "evaluate_manual":
        provider_name = "manual_talkie_csv"
    elif args.mode == "prepare_manual":
        provider_name = "manual_sheet_preparation"
    elif args.mode == "rewrite_only":
        provider_name = "rewrite_only_no_downstream"

    try:
        if args.mode in {"simulate_grid", "run_web_grid"}:
            selected_hparams, grid_rows = run_autoencoder_grid_search(
                dataset=dataset,
                modern_dictionary=modern_dictionary,
                primitive_dictionary=primitive_dictionary,
                config=config,
                client=client,
                out_dir=out_dir,
                grid_eval_items=args.grid_eval_items,
            )

        generator = build_generator(
            config,
            modern_dictionary,
            primitive_dictionary,
            dataset,
            autoencoder_hparams=selected_hparams,
        )

        for item in dataset:
            baseline_prompt = build_mcq_prompt(item["original_question"], item["choices"])
            rewrite_result = generator.generate(item)
            rewrite_cache_rows.append(asdict(rewrite_result))
            proposed_prompt = build_mcq_prompt(rewrite_result.rewritten_question, item["choices"])

            baseline_response = {"raw_response": ""}
            proposed_response = {"raw_response": ""}

            if args.mode in {"simulate", "simulate_grid"}:
                baseline_response = client.ask(
                    baseline_prompt,
                    item_id=item["id"],
                    condition="baseline",
                    item=item,
                    rewrite_result=rewrite_result,
                )
                proposed_response = client.ask(
                    proposed_prompt,
                    item_id=item["id"],
                    condition="proposed",
                    item=item,
                    rewrite_result=rewrite_result,
                )
            elif args.mode in {"run_web", "run_web_grid"}:
                baseline_response = client.ask(
                    baseline_prompt,
                    item_id=item["id"],
                    condition="baseline",
                    item=item,
                    rewrite_result=rewrite_result,
                )
                proposed_response = client.ask(
                    proposed_prompt,
                    item_id=item["id"],
                    condition="proposed",
                    item=item,
                    rewrite_result=rewrite_result,
                )
            elif args.mode == "evaluate_manual":
                baseline_response = {
                    "item_id": item["id"],
                    "condition": "baseline",
                    "prompt": baseline_prompt,
                    "raw_response": manual_responses.get((item["id"], "baseline"), ""),
                    "provider": "manual_talkie_csv",
                }
                proposed_response = {
                    "item_id": item["id"],
                    "condition": "proposed",
                    "prompt": proposed_prompt,
                    "raw_response": manual_responses.get((item["id"], "proposed"), ""),
                    "provider": "manual_talkie_csv",
                }

            row = build_result_row(
                item=item,
                rewrite_result=rewrite_result,
                baseline_prompt=baseline_prompt,
                proposed_prompt=proposed_prompt,
                baseline_response=baseline_response,
                proposed_response=proposed_response,
            )
            prepared_rows.append(row)
            if args.mode in {"simulate", "simulate_grid", "run_web", "run_web_grid", "evaluate_manual"}:
                result_rows.append(row)
    finally:
        if client_context is not None:
            client_context.__exit__(None, None, None)

    write_jsonl(cache_dir / "rewrite_cache.jsonl", rewrite_cache_rows)
    write_input_data_artifacts(input_dir, dataset, prepared_rows)

    if args.mode in {"prepare_manual", "rewrite_only"}:
        write_csv_dicts(out_dir / "prepared_prompts.csv", prepared_rows)
        write_jsonl(out_dir / "prepared_prompts.jsonl", prepared_rows)
        component_metrics = compute_component_metrics(prepared_rows)
        write_csv_dicts(out_dir / "component_metrics.csv", component_metrics, ["component", "metric", "value"])
        write_json(out_dir / "selected_autoencoder_hyperparameters.json", asdict(selected_hparams))
        if args.mode == "prepare_manual":
            prepare_manual_sheet(prepared_rows, out_dir)
        print(f"Wrote prepared prompts to {out_dir}")
        if args.mode == "prepare_manual":
            print(f"Manual Talkie sheet: {out_dir / 'manual_talkie_input_sheet.csv'}")
        return

    write_csv_dicts(out_dir / "per_item_results.csv", result_rows)
    write_jsonl(out_dir / "per_item_results.jsonl", result_rows)
    final_metrics, _detail = compute_downstream_metrics(result_rows)
    component_metrics = compute_component_metrics(result_rows)
    mcnemar = compute_mcnemar(result_rows)
    write_csv_dicts(out_dir / "final_metrics.csv", final_metrics, ["method", "accuracy", "macro_f1", "invalid_rate", "n_items"])
    write_csv_dicts(out_dir / "component_metrics.csv", component_metrics, ["component", "metric", "value"])
    write_json(out_dir / "paired_test_mcnemar.json", mcnemar)
    if not (out_dir / "selected_autoencoder_hyperparameters.json").exists():
        write_json(out_dir / "selected_autoencoder_hyperparameters.json", asdict(selected_hparams))
    write_qualitative_examples(result_rows, out_dir / "qualitative_examples.md")
    save_figures(result_rows, final_metrics, component_metrics, out_dir)
    generate_report(
        result_rows,
        final_metrics,
        component_metrics,
        mcnemar,
        out_dir,
        args.mode,
        provider_name,
        selected_hparams=selected_hparams,
        grid_rows=grid_rows,
    )

    print(f"Wrote results to {out_dir}")
    print(
        "Accuracy baseline/proposed: "
        f"{metric_lookup(final_metrics, 'baseline_raw_prompt', 'accuracy'):.3f} / "
        f"{metric_lookup(final_metrics, 'proposed_era_neutral_prompt', 'accuracy'):.3f}"
    )
    print(f"Report: {out_dir / 'report.md'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end Era-Neutral Prompt Generator pipeline for Term Project Team 2."
    )
    parser.add_argument(
        "--mode",
        choices=[
            "simulate",
            "simulate_grid",
            "prepare_manual",
            "evaluate_manual",
            "run_web",
            "run_web_grid",
            "rewrite_only",
        ],
        default="simulate",
        help="run_web_grid is the final required mode: Talkie web + Autoencoder grid search + full 100 raw/proposed evaluation.",
    )
    parser.add_argument("--n_items", type=int, default=100)
    parser.add_argument("--target_year", type=int, default=1930)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max_repair_attempts", type=int, default=2)
    parser.add_argument("--out_dir", default="results")
    parser.add_argument("--manual_response_csv", default="results/manual_talkie_input_sheet.csv")
    parser.add_argument("--force_regenerate_data", action="store_true")
    parser.add_argument(
        "--grid_eval_items",
        type=int,
        default=20,
        help="Number of items used for Talkie-based Autoencoder hyperparameter grid search. Set to 100 for full-grid evaluation.",
    )
    parser.add_argument("--talkie_url", default="https://talkie-lm.com/chat")
    parser.add_argument("--talkie_wait_timeout_ms", type=int, default=45000)
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only used by --mode run_web.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
