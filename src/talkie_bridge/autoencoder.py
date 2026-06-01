"""Small dependency-free primitive bottleneck autoencoder."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from talkie_bridge.data_schema import DatasetItem
from talkie_bridge.detector import tokenize


@dataclass(frozen=True)
class AutoencoderConfig:
    latent_dim: int = 12
    epochs: int = 50
    learning_rate: float = 0.08
    noise_prob: float = 0.1
    l2: float = 0.0005
    decode_threshold: float = 0.18
    top_k_primitives: int = 2


@dataclass(frozen=True)
class AutoencoderSelection:
    enabled: bool
    fallback_reason: str
    selected_config: dict[str, Any]
    dev_primitive_f1: float
    train_items: int
    dev_items: int


class TextVectorizer:
    def __init__(self, *, max_features: int = 512) -> None:
        self.max_features = max_features
        self.vocab: dict[str, int] = {}
        self.inverse_vocab: list[str] = []

    def fit(self, texts: Sequence[str]) -> None:
        counts: dict[str, int] = {}
        for text in texts:
            for token in tokenize(text):
                if len(token) < 2:
                    continue
                counts[token] = counts.get(token, 0) + 1
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[: self.max_features]
        self.inverse_vocab = [token for token, _count in ordered]
        self.vocab = {token: index for index, token in enumerate(self.inverse_vocab)}

    def transform(self, text: str) -> list[float]:
        values = [0.0] * len(self.inverse_vocab)
        for token in tokenize(text):
            index = self.vocab.get(token)
            if index is not None:
                values[index] = 1.0
        return values


class DenoisingPrimitiveAutoencoder:
    def __init__(self, config: AutoencoderConfig, *, seed: int = 13) -> None:
        self.config = config
        self.seed = seed
        self.input_vectorizer = TextVectorizer()
        self.output_vectorizer = TextVectorizer()
        self.encoder: list[list[float]] = []
        self.decoder: list[list[float]] = []
        self.output_bias: list[float] = []
        self.enabled = False

    def fit(self, pairs: Sequence[tuple[str, str]]) -> None:
        if not pairs:
            return
        rng = random.Random(self.seed)
        self.input_vectorizer.fit([source for source, _target in pairs])
        self.output_vectorizer.fit([target for _source, target in pairs])
        in_dim = len(self.input_vectorizer.inverse_vocab)
        out_dim = len(self.output_vectorizer.inverse_vocab)
        if not in_dim or not out_dim:
            return

        scale = 0.05
        self.encoder = [[rng.uniform(-scale, scale) for _ in range(self.config.latent_dim)] for _ in range(in_dim)]
        self.decoder = [[rng.uniform(-scale, scale) for _ in range(out_dim)] for _ in range(self.config.latent_dim)]
        self.output_bias = [0.0] * out_dim

        examples = [
            (self.input_vectorizer.transform(source), self.output_vectorizer.transform(target))
            for source, target in pairs
        ]
        for _epoch in range(self.config.epochs):
            rng.shuffle(examples)
            for x, y in examples:
                noisy_x = [0.0 if value and rng.random() < self.config.noise_prob else value for value in x]
                self._train_one(noisy_x, y)
        self.enabled = True

    def primitive_scores(self, text: str, primitive_dictionary: dict[str, dict[str, Any]]) -> list[tuple[str, float]]:
        if not self.enabled:
            return []
        token_scores = self.decode_token_scores(text)
        primitive_scores: list[tuple[str, float]] = []
        for primitive_id, meta in primitive_dictionary.items():
            phrase_tokens = [token for token in tokenize(str(meta.get("primitive_phrase", ""))) if len(token) > 3]
            if not phrase_tokens:
                continue
            score = sum(token_scores.get(token, 0.0) for token in phrase_tokens) / len(phrase_tokens)
            primitive_scores.append((primitive_id, score))
        return sorted(primitive_scores, key=lambda item: (-item[1], item[0]))

    def decode_token_scores(self, text: str) -> dict[str, float]:
        x = self.input_vectorizer.transform(text)
        z = self._encode(x)
        logits = self._decode_logits(z)
        return {
            token: _sigmoid(logit)
            for token, logit in zip(self.output_vectorizer.inverse_vocab, logits)
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "seed": self.seed,
            "enabled": self.enabled,
            "input_vocab": self.input_vectorizer.inverse_vocab,
            "output_vocab": self.output_vectorizer.inverse_vocab,
            "encoder": self.encoder,
            "decoder": self.decoder,
            "output_bias": self.output_bias,
        }

    def _train_one(self, x: list[float], y: list[float]) -> None:
        z = self._encode(x)
        logits = self._decode_logits(z)
        pred = [_sigmoid(value) for value in logits]
        error = [pred_j - y_j for pred_j, y_j in zip(pred, y)]

        old_decoder = [row[:] for row in self.decoder]
        for latent_index in range(self.config.latent_dim):
            for out_index in range(len(self.output_bias)):
                grad = error[out_index] * z[latent_index] + self.config.l2 * self.decoder[latent_index][out_index]
                self.decoder[latent_index][out_index] -= self.config.learning_rate * grad
        for out_index in range(len(self.output_bias)):
            self.output_bias[out_index] -= self.config.learning_rate * error[out_index]

        dz: list[float] = []
        for latent_index in range(self.config.latent_dim):
            backprop = sum(error[out_index] * old_decoder[latent_index][out_index] for out_index in range(len(error)))
            dz.append(backprop * z[latent_index] * (1.0 - z[latent_index]))

        for in_index, x_value in enumerate(x):
            if not x_value:
                continue
            for latent_index in range(self.config.latent_dim):
                grad = dz[latent_index] * x_value + self.config.l2 * self.encoder[in_index][latent_index]
                self.encoder[in_index][latent_index] -= self.config.learning_rate * grad

    def _encode(self, x: list[float]) -> list[float]:
        z: list[float] = []
        for latent_index in range(self.config.latent_dim):
            value = 0.0
            for in_index, x_value in enumerate(x):
                if x_value:
                    value += x_value * self.encoder[in_index][latent_index]
            z.append(_sigmoid(value))
        return z

    def _decode_logits(self, z: list[float]) -> list[float]:
        logits: list[float] = []
        for out_index, bias in enumerate(self.output_bias):
            value = bias
            for latent_index, z_value in enumerate(z):
                value += z_value * self.decoder[latent_index][out_index]
            logits.append(value)
        return logits


def train_select_autoencoder(
    items: Sequence[DatasetItem],
    primitive_dictionary: dict[str, dict[str, Any]],
    *,
    seed: int = 13,
) -> tuple[DenoisingPrimitiveAutoencoder | None, AutoencoderSelection]:
    train_items = [item for item in items if item.split == "train"]
    dev_items = [item for item in items if item.split == "dev"]
    if len(train_items) < 4 or not dev_items:
        return None, AutoencoderSelection(
            enabled=False,
            fallback_reason="need_at_least_4_train_items_and_1_dev_item",
            selected_config={},
            dev_primitive_f1=0.0,
            train_items=len(train_items),
            dev_items=len(dev_items),
        )

    configs = [
        AutoencoderConfig(latent_dim=8, epochs=40, learning_rate=0.08, noise_prob=0.1),
        AutoencoderConfig(latent_dim=16, epochs=60, learning_rate=0.06, noise_prob=0.15),
    ]
    train_pairs = build_training_pairs(train_items, primitive_dictionary)
    best_model: DenoisingPrimitiveAutoencoder | None = None
    best_score = -1.0
    best_config = configs[0]
    for index, config in enumerate(configs):
        model = DenoisingPrimitiveAutoencoder(config, seed=seed + index)
        model.fit(train_pairs)
        score = evaluate_primitive_f1(model, dev_items, primitive_dictionary)
        if score > best_score:
            best_score = score
            best_model = model
            best_config = config

    return best_model, AutoencoderSelection(
        enabled=best_model is not None and best_model.enabled,
        fallback_reason="",
        selected_config=asdict(best_config),
        dev_primitive_f1=max(0.0, best_score),
        train_items=len(train_items),
        dev_items=len(dev_items),
    )


def build_training_pairs(
    items: Sequence[DatasetItem],
    primitive_dictionary: dict[str, dict[str, Any]],
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for item in items:
        phrases = [
            str(primitive_dictionary.get(primitive_id, {}).get("primitive_phrase", ""))
            for primitive_id in item.required_primitives
        ]
        target = " ".join(phrase for phrase in phrases if phrase) or item.primitive_phrase
        if target:
            pairs.append((item.original_question, target))
    return pairs


def evaluate_primitive_f1(
    model: DenoisingPrimitiveAutoencoder,
    items: Sequence[DatasetItem],
    primitive_dictionary: dict[str, dict[str, Any]],
) -> float:
    if not items or not model.enabled:
        return 0.0
    f1s: list[float] = []
    for item in items:
        predicted = {
            primitive_id
            for primitive_id, score in model.primitive_scores(item.original_question, primitive_dictionary)[: model.config.top_k_primitives]
            if score >= model.config.decode_threshold
        }
        gold = set(item.required_primitives)
        tp = len(predicted & gold)
        precision = tp / len(predicted) if predicted else 0.0
        recall = tp / len(gold) if gold else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(f1s) / len(f1s)


def _sigmoid(value: float) -> float:
    if value < -40:
        return 0.0
    if value > 40:
        return 1.0
    return 1.0 / (1.0 + math.exp(-value))

