"""Didaktischer Starter-Code fuer einen RL-Workshop mit CarRacing.

Die Datei ist absichtlich linear und stark dokumentiert aufgebaut.
Sie soll wie ein kleiner roter Faden gelesen werden koennen:

1. Konfiguration
2. Aktionsabbildung
3. Vorverarbeitung
4. Modell
5. Datensammlung
6. REINFORCE-Update
7. Training und Evaluation

Wichtige didaktische Entscheidung:
Wir verwenden CarRacing mit kontinuierlichem Umgebungsformat,
aber mappen intern auf eine kleine diskrete Aktionsmenge.
Dadurch bleibt die Policy als kategoriale Verteilung leicht verstaendlich.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Categorical


# Vorverarbeitungskonstanten fuer den Workshop
PROCESSED_HEIGHT = 96
PROCESSED_WIDTH = 96
FRAME_STACK_SIZE = 32


_frame_stack_buffer: deque[Tensor] = deque(maxlen=FRAME_STACK_SIZE)


def reset_preprocess_state() -> None:
    """Leert den internen Frame-Stack zu Episodenbeginn."""

    _frame_stack_buffer.clear()


@dataclass(slots=True)
class WorkshopConfig:
    """Zentrale Konfiguration fuer Training und Evaluation.

    Alle haeufig genutzten Parameter stehen hier gesammelt, damit man im
    Workshop nicht ueber viele Stellen im Code springen muss.
    """

    env_id: str = "CarRacing-v3"
    episodes: int = 30
    max_steps_per_episode: int = 600
    gamma: float = 0.99
    learning_rate: float = 1e-4
    action_repeat: int = 4
    seed: int = 42
    hidden_dim: int = 256
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
    render_mode: str | None = "rgb_array"
    domain_randomize: bool = False
    save_plot_path: str = "training_rewards.png"
    model_path: str = "car_racing_policy.pt"


@dataclass(slots=True)
class EpisodeBatch:
    """Speichert alles, was fuer ein einzelnes Policy-Update noetig ist."""

    log_probs: list[Tensor]
    rewards: list[float]
    entropies: list[Tensor]
    attention_mean_weights: list[float]
    episode_reward: float
    steps: int


class DiscreteActionMapper:
    """Mappt diskrete Aktions-IDs auf das CarRacing-Aktionsformat.

    CarRacing erwartet drei kontinuierliche Werte:

    - steering in [-1, 1]
    - gas in [0, 1]
    - brake in [0, 1]

    Fuer den Workshop reduzieren wir das auf wenige handliche Aktionen.
    Die Liste ist absichtlich leicht veraenderbar, damit Teilnehmende sie in
    einer spaeteren Aufgabe erweitern koennen.
    """

    def __init__(self) -> None:
        self._actions = [
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.6, 0.0, 0.0], dtype=np.float32),
            np.array([-0.6, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.7, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, 0.8], dtype=np.float32),
        ]

    @property
    def size(self) -> int:
        return len(self._actions)

    def __getitem__(self, index: int) -> np.ndarray:
        return self._actions[index]

    def describe(self) -> Iterable[str]:
        labels = [
            "0 = nichts tun",
            "1 = rechts lenken",
            "2 = links lenken",
            "3 = Gas geben",
            "4 = bremsen",
        ]
        return labels


def set_global_seed(seed: int) -> None:
    """Setzt die wichtigsten Zufallsquellen fuer reproduzierbarere Runs."""

    np.random.seed(seed)
    torch.manual_seed(seed)


def preprocess_observation(observation: np.ndarray, device: torch.device) -> Tensor:
    """Konvertiert ein RGB-Bild in einen normierten Tensor.

    Eingang:
    - NumPy-Array mit Form H x W x C und uint8-Werten in [0, 255]

    Ausgang:
    - Tensor mit Form 1 x C x H x W und float32-Werten in [0, 1]

    Diese Funktion ist ein didaktischer Erweiterungspunkt.
    Hier koennen Teilnehmende spaeter sehr gezielt experimentieren:

    - Graustufen statt RGB
    - Zuschneiden der unteren Anzeige
    - Downsampling
    - Frame-Stacking in einer Wrapper-Struktur
    """

    # 1) RGB -> Graustufen, um die Eingabe zu vereinfachen.
    rgb = observation.astype(np.float32) / 255.0
    grayscale = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]

    processed = torch.from_numpy(grayscale).unsqueeze(0).unsqueeze(0).to(device)

    # 2) Frame-Stacking fuer Bewegungsinformation.
    # Beim ersten Schritt fuellen wir den Puffer mit identischen Frames auf.
    current_frame = processed.squeeze(0)
    if not _frame_stack_buffer:
        for _ in range(FRAME_STACK_SIZE):
            _frame_stack_buffer.append(current_frame)
    else:
        _frame_stack_buffer.append(current_frame)

    stacked = torch.cat(list(_frame_stack_buffer), dim=0).unsqueeze(0)
    return stacked


class FrameFeatureAttention(nn.Module):
    """Gewichtet die Frames im Stack anhand gelernter Aufmerksamkeitswerte."""

    def __init__(self, frame_count: int, hidden_dim: int = 16) -> None:
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Linear(frame_count, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, frame_count),
        )

    def forward(self, frame_stack: Tensor) -> tuple[Tensor, Tensor]:
        if frame_stack.ndim != 4:
            raise ValueError(
                f"FrameFeatureAttention erwartet BxTxHxW, erhalten: {tuple(frame_stack.shape)}"
            )

        frame_summary = frame_stack.mean(dim=(2, 3))
        logits = self.score_net(frame_summary)
        weights = torch.softmax(logits, dim=1)
        weighted_stack = frame_stack * weights.unsqueeze(-1).unsqueeze(-1)
        return weighted_stack, weights


class PolicyNetwork(nn.Module):
    """Kleines CNN fuer eine diskrete Policy auf Bildbeobachtungen.

    Das Modell gibt Logits aus. Aus diesen Logits erzeugen wir spaeter eine
    kategoriale Verteilung, aus der eine diskrete Aktion gesampelt wird.
    """

    def __init__(self, action_count: int, hidden_dim: int) -> None:
        super().__init__()
        self.frame_attention = FrameFeatureAttention(frame_count=FRAME_STACK_SIZE)
        self.last_attention_weights: Tensor | None = None
        self.encoder = nn.Sequential(
            nn.Conv2d(FRAME_STACK_SIZE, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        encoder_out = self._infer_encoder_size()

        self.policy_head = nn.Sequential(
            nn.Linear(encoder_out, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_count),
        )

    def _infer_encoder_size(self) -> int:
        dummy = torch.zeros(1, FRAME_STACK_SIZE, PROCESSED_HEIGHT, PROCESSED_WIDTH)
        with torch.no_grad():
            return int(self.encoder(dummy).shape[1])

    def forward(self, observation: Tensor) -> Tensor:
        attended, weights = self.frame_attention(observation)
        self.last_attention_weights = weights.detach()
        encoded = self.encoder(attended)
        return self.policy_head(encoded)


def discounted_returns(rewards: list[float], gamma: float, device: torch.device) -> Tensor:
    """Berechnet diskontierte Returns fuer eine Episode.

    Fuer jeden Zeitpunkt t berechnen wir:

    G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...
    """

    running_return = 0.0
    returns: list[float] = []

    for reward in reversed(rewards):
        running_return = reward + gamma * running_return
        returns.append(running_return)

    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32, device=device)


def choose_action(policy: PolicyNetwork, observation: Tensor) -> tuple[int, Tensor, Tensor]:
    """Fuehrt einen Forward-Pass aus und sampelt eine Aktion.

    Rueckgabe:
    - diskrete Aktions-ID
    - Log-Wahrscheinlichkeit der gewaelten Aktion
    - Entropie der Verteilung
    """

    logits = policy(observation)
    distribution = Categorical(logits=logits)
    action = distribution.sample()
    log_prob = distribution.log_prob(action)
    entropy = distribution.entropy()
    return int(action.item()), log_prob.squeeze(), entropy.squeeze()


def run_episode(
    env: gym.Env,
    policy: PolicyNetwork,
    action_mapper: DiscreteActionMapper,
    config: WorkshopConfig,
    device: torch.device,
    episode_seed: int | None = None,
) -> EpisodeBatch:
    """Sammelt eine komplette Episode in der Umgebung.

    Hier ist die Trennung zwischen Umgebungsinteraktion und Lernschritt wichtig.
    Zuerst sammeln wir nur Daten. Das eigentliche Update passiert danach in einer
    separaten Funktion.
    """

    observation, _ = env.reset(seed=episode_seed)
    reset_preprocess_state()
    log_probs: list[Tensor] = []
    rewards: list[float] = []
    entropies: list[Tensor] = []
    attention_weights: list[Tensor] = []
    episode_reward = 0.0
    steps = 0

    while steps < config.max_steps_per_episode:
        state = preprocess_observation(observation, device)
        action_index, log_prob, entropy = choose_action(policy, state)
        action = action_mapper[action_index]

        if policy.last_attention_weights is not None:
            attention_weights.append(policy.last_attention_weights.squeeze(0))

        repeated_reward = 0.0
        terminated = False
        truncated = False

        for _ in range(config.action_repeat):
            observation, reward, terminated, truncated, _ = env.step(action)
            repeated_reward += reward
            steps += 1

            if terminated or truncated or steps >= config.max_steps_per_episode:
                break

        log_probs.append(log_prob)
        rewards.append(repeated_reward)
        entropies.append(entropy)
        episode_reward += repeated_reward

        if terminated or truncated:
            break

    if attention_weights:
        attention_mean = torch.stack(attention_weights).mean(dim=0)
        attention_mean_weights = [float(value) for value in attention_mean.tolist()]
    else:
        attention_mean_weights = [0.0 for _ in range(FRAME_STACK_SIZE)]

    return EpisodeBatch(
        log_probs=log_probs,
        rewards=rewards,
        entropies=entropies,
        attention_mean_weights=attention_mean_weights,
        episode_reward=episode_reward,
        steps=steps,
    )


def update_policy_from_episode(
    episode: EpisodeBatch,
    policy: PolicyNetwork,
    optimizer: torch.optim.Optimizer,
    config: WorkshopConfig,
    device: torch.device,
) -> float:
    """Fuehrt genau einen REINFORCE-Update-Schritt aus.

    Der Loss lautet in seiner einfachen Form:

    loss = - sum(log_prob_t * G_t)

    Typische Workshop-Erweiterungen:

    - Returns normalisieren
    - Entropy-Bonus einbauen
    - Baseline oder Value-Netz hinzufuegen
    """

    returns = discounted_returns(episode.rewards, config.gamma, device)

    # TODO fuer den Workshop:
    # Aktiviert als Experiment die Return-Normalisierung, um die Varianz zu
    # reduzieren. Eine moegliche Variante ist:
    # returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    log_probs = torch.stack(episode.log_probs)
    entropies = torch.stack(episode.entropies)

    policy_loss = -(log_probs * returns).sum()

    # TODO fuer den Workshop:
    # Fuegt testweise einen Entropy-Bonus hinzu, etwa so:
    # entropy_bonus = entropies.sum()
    # loss = policy_loss - 0.01 * entropy_bonus
    loss = policy_loss

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
    optimizer.step()

    return float(loss.item())


def make_env(config: WorkshopConfig, render_mode: str | None = None) -> gym.Env:
    """Erzeugt die CarRacing-Umgebung mit den fuer den Workshop relevanten Parametern."""

    return gym.make(
        config.env_id,
        continuous=True,
        render_mode=render_mode,
        domain_randomize=config.domain_randomize,
    )


def train(config: WorkshopConfig) -> list[float]:
    """Trainiert die Policy fuer eine feste Anzahl Episoden."""

    set_global_seed(config.seed)
    device = torch.device(config.device)
    env = make_env(config, render_mode=config.render_mode)
    action_mapper = DiscreteActionMapper()

    print("Verfuegbare diskrete Aktionen:")
    for label in action_mapper.describe():
        print(f"  {label}")

    policy = PolicyNetwork(action_count=action_mapper.size, hidden_dim=config.hidden_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    reward_history: list[float] = []

    try:
        for episode_index in range(1, config.episodes + 1):
            episode = run_episode(
                env,
                policy,
                action_mapper,
                config,
                device,
                episode_seed=config.seed + episode_index - 1,
            )
            loss_value = update_policy_from_episode(episode, policy, optimizer, config, device)
            reward_history.append(episode.episode_reward)

            mean_recent = float(np.mean(reward_history[-10:]))
            attention_str = " | ".join(
                f"F{i}:{weight:.3f}" for i, weight in enumerate(episode.attention_mean_weights)
            )
            print(
                f"Episode {episode_index:03d} | "
                f"Reward: {episode.episode_reward:8.2f} | "
                f"Schritte: {episode.steps:4d} | "
                f"Loss: {loss_value:10.2f} | "
                f"Mittel letzte 10: {mean_recent:8.2f} | "
                f"Attention: {attention_str}"
            )
    finally:
        env.close()

    save_model(policy, config.model_path)
    print(f"Modell gespeichert unter: {config.model_path}")

    return reward_history


def evaluate(config: WorkshopConfig) -> None:
    """Fuehrt einige Episoden ohne Lernschritt aus und zeigt Rewards an."""

    set_global_seed(config.seed)
    device = torch.device(config.device)
    env = make_env(config, render_mode=config.render_mode)
    action_mapper = DiscreteActionMapper()
    policy = PolicyNetwork(action_count=action_mapper.size, hidden_dim=config.hidden_dim).to(device)
    load_model_if_available(policy, config.model_path, device)

    try:
        for episode_index in range(1, config.episodes + 1):
            episode = run_episode(
                env,
                policy,
                action_mapper,
                config,
                device,
                episode_seed=config.seed + episode_index - 1,
            )
            attention_str = " | ".join(
                f"F{i}:{weight:.3f}" for i, weight in enumerate(episode.attention_mean_weights)
            )
            print(
                f"Evaluation Episode {episode_index:03d} | "
                f"Reward: {episode.episode_reward:8.2f} | "
                f"Schritte: {episode.steps:4d} | "
                f"Attention: {attention_str}"
            )
    finally:
        env.close()


def save_reward_plot(rewards: list[float], output_path: str) -> Path:
    """Speichert eine einfache Reward-Kurve fuer spaetere Diskussionen."""

    path = Path(output_path)
    plt.figure(figsize=(10, 4))
    plt.plot(rewards, label="episodischer Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("CarRacing Workshop: Trainingsverlauf")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def save_model(policy: PolicyNetwork, output_path: str) -> Path:
    """Speichert die Modellgewichte fuer spaetere Evaluationen."""

    path = Path(output_path)
    torch.save(policy.state_dict(), path)
    return path


def load_model_if_available(policy: PolicyNetwork, model_path: str, device: torch.device) -> None:
    """Laedt vorhandene Modellgewichte oder gibt einen klaren Hinweis aus."""

    path = Path(model_path)

    if not path.exists():
        print(
            "Kein gespeichertes Modell gefunden. Evaluation laeuft deshalb mit "
            "zufaellig initialisierten Gewichten."
        )
        return

    state_dict = torch.load(path, map_location=device)
    policy.load_state_dict(state_dict)
    print(f"Modell geladen aus: {path}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Definiert eine kleine CLI, damit Experimente leicht gestartet werden koennen."""

    parser = argparse.ArgumentParser(description="REINFORCE-Starter fuer CarRacing")
    parser.add_argument("--mode", choices=["train", "evaluate"], default="train")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--action-repeat", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", choices=["none", "human", "rgb_array"], default="rgb_array")
    parser.add_argument("--domain-randomize", action="store_true")
    parser.add_argument("--save-plot", default="training_rewards.png")
    parser.add_argument("--model-path", default="car_racing_policy.pt")
    return parser


def config_from_args(args: argparse.Namespace) -> WorkshopConfig:
    """Baut aus CLI-Argumenten eine klar benannte Konfiguration."""

    render_mode = None if args.render == "none" else args.render
    return WorkshopConfig(
        episodes=args.episodes,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        action_repeat=args.action_repeat,
        seed=args.seed,
        render_mode=render_mode,
        domain_randomize=args.domain_randomize,
        save_plot_path=args.save_plot,
        model_path=args.model_path,
    )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = config_from_args(args)

    if args.mode == "train":
        rewards = train(config)
        plot_path = save_reward_plot(rewards, config.save_plot_path)
        print(f"Reward-Kurve gespeichert unter: {plot_path}")
    else:
        evaluate(config)


if __name__ == "__main__":
    main()