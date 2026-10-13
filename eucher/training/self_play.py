"""Self-play training system for ML models."""

import multiprocessing
import os
import queue
import uuid
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypedDict

from tqdm import tqdm

from eucher.game import Game
from eucher.training.data_collector import GameDataCollector
from eucher.training.profiling import timed_operation

# One thread per worker. Numeric libraries otherwise start a pool per process.
_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

_worker_trainer: Optional["SelfPlayTrainer"] = None
_worker_progress: Any = None


class CollectedGameData(TypedDict):
    """Decision records and outcomes from one batch of simulated games."""

    order_up: List[Dict[str, Any]]
    call_trump: List[Dict[str, Any]]
    play_card: List[Dict[str, Any]]
    discard: List[Dict[str, Any]]
    outcomes: List[Dict[str, Any]]
    replays: List[str]


def resolve_worker_count(workers: int, total_games: int) -> int:
    """
    Choose how many processes to use for a batch of games.

    Parameters
    ----------
    workers : int
        Requested process count. ``0`` means one process per CPU core.
    total_games : int
        Number of games in the batch. A non-empty batch never uses more
        processes than games.

    Returns
    -------
    int
        Process count of at least 1.

    Raises
    ------
    ValueError
        If ``workers`` is negative.
    """
    if workers < 0:
        raise ValueError(f"workers must be >= 0, got {workers}")
    if workers == 0:
        requested = os.cpu_count() or 1
    else:
        requested = workers
    if total_games < 1:
        return 1
    return max(1, min(requested, total_games))


def _push_single_thread_env() -> Dict[str, Optional[str]]:
    """
    Limit numeric libraries to one thread in processes spawned after this call.

    Returns
    -------
    Dict[str, Optional[str]]
        Previous values so the caller can restore this process environment.
    """
    previous: Dict[str, Optional[str]] = {}
    for name in _THREAD_ENV_VARS:
        previous[name] = os.environ.get(name)
        os.environ[name] = "1"
    return previous


def _restore_env(previous: Dict[str, Optional[str]]) -> None:
    """
    Restore environment variables captured by :func:`_push_single_thread_env`.

    Parameters
    ----------
    previous : Dict[str, Optional[str]]
        Earlier values. ``None`` means the variable was unset.
    """
    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _init_game_worker(output_dir: str, progress_queue: Any) -> None:
    """
    Build the trainer used for every game this worker process plays.

    Parameters
    ----------
    output_dir : str
        Directory where this worker writes game replays and datasets.
    progress_queue : Any
        Queue that receives ``1`` after each finished game.
    """
    global _worker_progress, _worker_trainer
    _worker_progress = progress_queue
    _worker_trainer = SelfPlayTrainer(output_dir=Path(output_dir))


def _take_collected_data(collector: GameDataCollector, keep_decisions: bool) -> CollectedGameData:
    """
    Copy collector records and clear the collector.

    Parameters
    ----------
    collector : GameDataCollector
        Collector that just finished a batch of games.
    keep_decisions : bool
        When False, decision rows are dropped so they are not sent back to
        the parent after the worker has already written them to disk.

    Returns
    -------
    CollectedGameData
        Records the parent should merge.
    """
    collected: CollectedGameData = {
        "order_up": list(collector.order_up_data) if keep_decisions else [],
        "call_trump": list(collector.call_trump_data) if keep_decisions else [],
        "play_card": list(collector.play_card_data) if keep_decisions else [],
        "discard": list(collector.discard_data) if keep_decisions else [],
        "outcomes": list(collector.game_outcomes),
        "replays": list(collector.saved_game_replays),
    }
    collector.clear_data()
    collector.saved_game_replays.clear()
    return collected


def _play_collected_games(
    player_configs: List[List[Tuple[str, str]]],
    save_data: bool,
) -> CollectedGameData:
    """
    Play a batch of games inside a worker process.

    Parameters
    ----------
    player_configs : List[List[Tuple[str, str]]]
        One player configuration per game.
    save_data : bool
        When True, write this batch's datasets before returning.

    Returns
    -------
    CollectedGameData
        Outcomes, replay ids, and decision rows when they were not saved here.

    Raises
    ------
    RuntimeError
        If the worker initializer did not run.
    """
    if _worker_trainer is None or _worker_progress is None:
        raise RuntimeError("Game worker was not initialized")
    collector = _worker_trainer.data_collector
    collector.clear_data()
    collector.saved_game_replays.clear()
    for player_config in player_configs:
        _worker_trainer.run_game_with_collection(player_config)
        _worker_progress.put(1)
    if save_data:
        collector.save_data()
    return _take_collected_data(collector, keep_decisions=not save_data)


def _split_jobs(
    jobs: Sequence[List[Tuple[str, str]]],
    workers: int,
) -> List[List[List[Tuple[str, str]]]]:
    """
    Deal games across workers in round-robin order.

    Parameters
    ----------
    jobs : Sequence[List[Tuple[str, str]]]
        Player configuration for each game.
    workers : int
        Number of non-empty chunks to produce.

    Returns
    -------
    List[List[List[Tuple[str, str]]]]
        One chunk of games per worker.
    """
    chunks: List[List[List[Tuple[str, str]]]] = [[] for _ in range(workers)]
    for index, job in enumerate(jobs):
        chunks[index % workers].append(job)
    return [chunk for chunk in chunks if chunk]


def _drain_progress(
    progress_queue: Any,
    progress: Optional[tqdm],
    completed: int,
    total_games: int,
) -> int:
    """
    Apply finished-game messages already waiting on the progress queue.

    Parameters
    ----------
    progress_queue : Any
        Queue fed by worker processes.
    progress : Optional[tqdm]
        Progress bar to advance. Prints every tenth game when omitted.
    completed : int
        Games already counted.
    total_games : int
        Games expected for the whole round.

    Returns
    -------
    int
        Updated completed-game count.
    """
    while True:
        try:
            progress_queue.get_nowait()
        except queue.Empty:
            return completed
        completed += 1
        if progress is not None:
            progress.update(1)
        elif completed % 10 == 0 or completed == total_games:
            print(f"  Game {completed}/{total_games}")


def _track_progress(
    futures: Sequence[Future[CollectedGameData]],
    progress_queue: Any,
    progress: Optional[tqdm],
    total_games: int,
) -> int:
    """
    Update progress until every worker batch has finished.

    Parameters
    ----------
    futures : Sequence[Future[CollectedGameData]]
        One future per worker batch.
    progress_queue : Any
        Queue fed by worker processes.
    progress : Optional[tqdm]
        Progress bar to advance. Prints every tenth game when omitted.
    total_games : int
        Games expected for the whole round.

    Returns
    -------
    int
        Games observed on the progress queue.
    """
    completed = 0
    pending = set(futures)
    while pending:
        completed = _drain_progress(progress_queue, progress, completed, total_games)
        done, pending = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
        for future in done:
            future.result()
    return _drain_progress(progress_queue, progress, completed, total_games)


class SelfPlayTrainer:
    """Trains ML models through self-play with various player combinations."""

    def __init__(
        self,
        data_collector: Optional[GameDataCollector] = None,
        output_dir: Optional[Path] = None,
    ) -> None:
        """
        Initialize the self-play trainer.

        Parameters
        ----------
        data_collector : Optional[GameDataCollector]
            Data collector instance. If None, creates a new one.
        output_dir : Optional[Path]
            Directory for output data. If None, uses default from MLConfig.
        """
        from plugins.ml.ml_config import MLConfig

        self.config = MLConfig()
        if data_collector is None:
            if output_dir is None:
                output_dir = self.config.training_data_dir
            data_collector = GameDataCollector(output_dir)
        self.data_collector = data_collector

    @timed_operation("SelfPlayTrainer.run_game_with_collection")
    def run_game_with_collection(
        self, player_config: List[Tuple[str, str]], collect_from_players: Optional[List[int]] = None
    ) -> None:
        """
        Run a game and collect training data.

        Parameters
        ----------
        player_config : List[Tuple[str, str]]
            List of (name, profile_type) tuples for each player.
        collect_from_players : Optional[List[int]]
            List of player IDs to collect data from. If None, collects from all computer players.
        """
        game_id = str(uuid.uuid4())
        self.data_collector.start_game(game_id)

        # Create game with the same UUID as game_id for consistency
        game = Game(player_config, seed=game_id, allow_untrained=True)

        # Determine which players to collect from
        if collect_from_players is None:
            collect_from_players = [
                i for i, (_, profile_type) in enumerate(player_config) if profile_type != "human"
            ]

        # Store original methods and create wrappers for data collection
        original_methods = {}
        for player_idx in collect_from_players:
            player = game.players[player_idx]
            original_methods[player_idx] = {
                "decide_order_up": player.profile.decide_order_up,
                "decide_call_trump": player.profile.decide_call_trump,
                "play_card": player.profile.play_card,
                "choose_card_to_discard": player.profile.choose_card_to_discard,
            }

            # Create wrapper methods that collect data
            def make_order_up_wrapper(pid: int):
                original = original_methods[pid]["decide_order_up"]

                def wrapper(p, tc, di, ts):
                    decision = original(p, tc, di, ts)
                    game_state = {
                        "trick_number": getattr(game, "_current_trick_number", 0),
                        "tricks_won_team0": getattr(game, "_current_tricks_won", [0, 0])[0],
                        "tricks_won_team1": getattr(game, "_current_tricks_won", [0, 0])[1],
                    }
                    self.data_collector.record_order_up_decision(
                        pid, p.hand, tc, di, decision, game_state
                    )
                    return decision

                return wrapper

            def make_call_trump_wrapper(pid: int):
                original = original_methods[pid]["decide_call_trump"]

                def wrapper(p, tc, ts, mc=False):
                    decision = original(p, tc, ts, mc)
                    game_state = {
                        "trick_number": getattr(game, "_current_trick_number", 0),
                        "tricks_won_team0": getattr(game, "_current_tricks_won", [0, 0])[0],
                        "tricks_won_team1": getattr(game, "_current_tricks_won", [0, 0])[1],
                    }
                    self.data_collector.record_call_trump_decision(
                        pid, p.hand, tc, decision, game_state
                    )
                    return decision

                return wrapper

            def make_play_card_wrapper(pid: int):
                original = original_methods[pid]["play_card"]

                def wrapper(p, ls, ts, tc, tpi):
                    decision = original(p, ls, ts, tc, tpi)

                    # Check if this play is a renege
                    from eucher.rules import RulesEngine

                    rules = RulesEngine()
                    is_renege = not rules.can_play_card(decision, p.hand, ls, ts)

                    game_state = {
                        "trick_number": getattr(game, "_current_trick_number", 0),
                        "tricks_won_team0": getattr(game, "_current_tricks_won", [0, 0])[0],
                        "tricks_won_team1": getattr(game, "_current_tricks_won", [0, 0])[1],
                        "is_renege": is_renege,  # Include renege flag for training
                    }
                    self.data_collector.record_play_card_decision(
                        pid, p.hand, ls, ts, tc, decision, game_state
                    )
                    return decision

                return wrapper

            def make_discard_wrapper(pid: int):
                original = original_methods[pid]["choose_card_to_discard"]

                def wrapper(player, turned_card=None, ordered_up_by=None):
                    decision = original(player, turned_card, ordered_up_by)
                    game_state = {
                        "trick_number": getattr(game, "_current_trick_number", 0),
                        "tricks_won_team0": getattr(game, "_current_tricks_won", [0, 0])[0],
                        "tricks_won_team1": getattr(game, "_current_tricks_won", [0, 0])[1],
                    }
                    self.data_collector.record_discard_decision(
                        pid, player.hand, decision, game_state
                    )
                    return decision

                return wrapper

            # Replace methods with wrappers
            player.profile.decide_order_up = make_order_up_wrapper(player_idx)
            player.profile.decide_call_trump = make_call_trump_wrapper(player_idx)
            player.profile.play_card = make_play_card_wrapper(player_idx)
            player.profile.choose_card_to_discard = make_discard_wrapper(player_idx)

        # Play game
        tricks_won_history: List[List[int]] = []
        while True:
            continue_game = game.play_hand()
            tricks_won = getattr(game, "_current_tricks_won", [0, 0])
            tricks_won_history.append(tricks_won.copy())

            winner = game.get_winner()

            if winner is not None or not continue_game:
                break

        # Record game outcome (including renege information)
        renege_occurred = getattr(game, "_renege_occurred", False)
        renege_player_id = getattr(game, "_renege_player_id", None)
        self.data_collector.record_game_outcome(
            game_id,
            game.get_scores(),
            tricks_won_history,
            winner,
            renege_occurred,
            renege_player_id,
        )

        # Save full game replay
        try:
            # Game already has game_uuid set from seed=game_id above
            from eucher.game_file import save_game

            save_game(game, self.data_collector.output_dir)
            self.data_collector.saved_game_replays.append(game_id)
        except Exception as e:
            # Don't fail if game replay save fails, but log it
            import sys

            print(f"Warning: Failed to save game replay for {game_id}: {e}", file=sys.stderr)

    @timed_operation("SelfPlayTrainer.run_training_round")
    def run_training_round(
        self,
        num_games: int = 100,
        player_combinations: Optional[List[List[Tuple[str, str]]]] = None,
        save_data: bool = True,
        show_progress: bool = True,
        data_prefix: Optional[str] = None,
        workers: int = 1,
    ) -> None:
        """
        Run a round of training games with various player combinations.

        Parameters
        ----------
        num_games : int
            Number of games to run per combination.
        player_combinations : Optional[List[List[Tuple[str, str]]]]
            List of player configurations to test. If None, uses default combinations.
        save_data : bool
            Whether to save collected data at the end. Set False for batched saves.
            With more than one worker and no ``data_prefix``, each worker writes
            its own UUID dataset files.
        show_progress : bool
            Whether to display a tqdm progress bar for game generation.
        data_prefix : Optional[str]
            If set, write ``{data_prefix}_<decision>.npz`` files. Otherwise write
            UUID ``.npz`` datasets with matching ``.txt`` metadata.
        workers : int
            Processes used to simulate games. ``0`` uses one process per CPU
            core. ``1`` runs games in this process.

        Raises
        ------
        ValueError
            If ``workers`` is negative.
        """
        if player_combinations is None:
            player_combinations = self._get_default_combinations()

        total_games = len(player_combinations) * num_games
        worker_count = resolve_worker_count(workers, total_games)
        # Named prefix files are one shared path per dataset, so the parent
        # writes them after workers return their rows.
        save_in_workers = save_data and data_prefix is None
        saved_by_workers = False
        if show_progress:
            if worker_count > 1:
                print(f"Worker processes: {worker_count}")
            with tqdm(total=total_games, desc="Generating games", unit="game") as progress:
                saved_by_workers = self._play_games(
                    player_combinations,
                    num_games,
                    progress,
                    worker_count,
                    save_in_workers,
                )
        else:
            if worker_count == 1:
                print(f"Running {len(player_combinations)} player combinations...")
            else:
                print(
                    f"Running {len(player_combinations)} player combinations "
                    f"on {worker_count} processes..."
                )
            saved_by_workers = self._play_games(
                player_combinations,
                num_games,
                None,
                worker_count,
                save_in_workers,
            )

        if save_data and not saved_by_workers:
            print("\nSaving collected data...")
            if data_prefix:
                self.data_collector.save_data(prefix=data_prefix, use_uuid_naming=False)
            else:
                self.data_collector.save_data()

    def _play_combinations(
        self,
        player_combinations: List[List[Tuple[str, str]]],
        num_games: int,
        progress: Optional[tqdm],
    ) -> None:
        """
        Play each player combination for the requested number of games.

        Parameters
        ----------
        player_combinations : List[List[Tuple[str, str]]]
            Player configurations to run.
        num_games : int
            Number of games to run per combination.
        progress : Optional[tqdm]
            Progress bar to update after each game. Prints status when omitted.
        """
        combination_count = len(player_combinations)
        for combo_idx, combo in enumerate(player_combinations):
            if progress is None:
                print(f"\nCombination {combo_idx + 1}/{combination_count}: {combo}")
            else:
                combo_label = "/".join(profile_type for _, profile_type in combo)
                progress.set_postfix_str(
                    f"combo {combo_idx + 1}/{combination_count} [{combo_label}]"
                )
            for game_num in range(num_games):
                if progress is None and (game_num + 1) % 10 == 0:
                    print(f"  Game {game_num + 1}/{num_games}")
                self.run_game_with_collection(combo)
                if progress is not None:
                    progress.update(1)

    def _play_games(
        self,
        player_combinations: List[List[Tuple[str, str]]],
        num_games: int,
        progress: Optional[tqdm],
        worker_count: int,
        save_data: bool,
    ) -> bool:
        """
        Play every requested game, in this process or across a process pool.

        Parameters
        ----------
        player_combinations : List[List[Tuple[str, str]]]
            Player configurations to run.
        num_games : int
            Number of games to run per combination.
        progress : Optional[tqdm]
            Progress bar to update after each game.
        worker_count : int
            Process count. ``1`` keeps the simulation in this process.
        save_data : bool
            Whether workers should write dataset files themselves.

        Returns
        -------
        bool
            True when worker processes already wrote the dataset files.
        """
        if worker_count <= 1:
            self._play_combinations(player_combinations, num_games, progress)
            return False
        self._play_combinations_parallel(
            player_combinations,
            num_games,
            progress,
            worker_count,
            save_data,
        )
        return save_data

    def _play_combinations_parallel(
        self,
        player_combinations: List[List[Tuple[str, str]]],
        num_games: int,
        progress: Optional[tqdm],
        worker_count: int,
        save_data: bool,
    ) -> None:
        """
        Simulate games on a process pool, one batch of games per worker.

        Parameters
        ----------
        player_combinations : List[List[Tuple[str, str]]]
            Player configurations to run.
        num_games : int
            Number of games to run per combination.
        progress : Optional[tqdm]
            Progress bar to update as workers finish games.
        worker_count : int
            Number of worker processes.
        save_data : bool
            Whether each worker writes its dataset files before returning.
        """
        jobs: List[List[Tuple[str, str]]] = []
        for combo in player_combinations:
            jobs.extend([combo] * num_games)
        chunks = _split_jobs(jobs, worker_count)
        if progress is not None:
            progress.set_postfix_str(f"{len(chunks)} workers")

        context = multiprocessing.get_context("spawn")
        progress_queue = context.Queue()
        previous_env = _push_single_thread_env()
        executor = ProcessPoolExecutor(
            max_workers=len(chunks),
            mp_context=context,
            initializer=_init_game_worker,
            initargs=(str(self.data_collector.output_dir), progress_queue),
        )
        try:
            futures = [executor.submit(_play_collected_games, chunk, save_data) for chunk in chunks]
            completed = _track_progress(
                futures,
                progress_queue,
                progress,
                len(jobs),
            )
            for future in futures:
                self._merge_collected_game(future.result())
            if progress is not None and completed < len(jobs):
                progress.update(len(jobs) - completed)
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
            progress_queue.close()
            progress_queue.join_thread()
            _restore_env(previous_env)

    def _merge_collected_game(self, collected: CollectedGameData) -> None:
        """
        Append one worker batch onto this trainer's collector.

        Parameters
        ----------
        collected : CollectedGameData
            Records returned by a worker process.
        """
        collector = self.data_collector
        collector.order_up_data.extend(collected["order_up"])
        collector.call_trump_data.extend(collected["call_trump"])
        collector.play_card_data.extend(collected["play_card"])
        collector.discard_data.extend(collected["discard"])
        collector.game_outcomes.extend(collected["outcomes"])
        collector.saved_game_replays.extend(collected["replays"])

    def _get_default_combinations(self) -> List[List[Tuple[str, str]]]:
        """
        Get default player combinations for training.

        Returns
        -------
        List[List[Tuple[str, str]]]
            List of player configurations.
        """
        return [
            # All ML players
            [
                ("ML1", "ml_sklearn"),
                ("ML2", "ml_sklearn"),
                ("ML3", "ml_sklearn"),
                ("ML4", "ml_sklearn"),
            ],
            # ML vs Random
            [
                ("ML1", "ml_sklearn"),
                ("Random1", "random"),
                ("ML2", "ml_sklearn"),
                ("Random2", "random"),
            ],
            # ML with AI partner vs Random opponents
            [
                ("ML1", "ml_sklearn"),
                ("AI1", "ai"),
                ("Random1", "random"),
                ("Random2", "random"),
            ],
            # ML with Random partner vs AI and Random opponents
            [
                ("ML1", "ml_sklearn"),
                ("Random1", "random"),
                ("AI1", "ai"),
                ("Random2", "random"),
            ],
            # ML vs Heuristic
            [
                ("ML1", "ml_sklearn"),
                ("Heuristic1", "heuristic"),
                ("ML2", "ml_sklearn"),
                ("Heuristic2", "heuristic"),
            ],
            # Mixed: ML, AI, Random, Heuristic
            [
                ("ML1", "ml_sklearn"),
                ("AI1", "ai"),
                ("Random1", "random"),
                ("Heuristic1", "heuristic"),
            ],
        ]


if __name__ == "__main__":
    trainer = SelfPlayTrainer()
    trainer.run_training_round(num_games=50)
