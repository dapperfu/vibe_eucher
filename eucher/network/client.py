"""Client implementation for network multiplayer Euchre."""

import socket
import sys
import time
from typing import Any, Callable, Dict, Optional

try:
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from eucher.network.protocol import (
    MessageType,
    DecisionType,
    create_message,
    create_decision_response,
    parse_decision_response,
    serialize_message,
    deserialize_message,
)


class EuchreClient:
    """Client for connecting to Euchre game server."""

    def __init__(
        self,
        server_host: str,
        server_port: int,
        timeout: int = 300,
    ) -> None:
        """
        Initialize the client.

        Parameters
        ----------
        server_host : str
            Server hostname or IP address.
        server_port : int
            Server port number.
        timeout : int
            Connection timeout in seconds (default: 300).
        """
        self.server_host = server_host
        self.server_port = server_port
        self.timeout = timeout
        self.socket: Optional[socket.socket] = None
        self.player_id: Optional[int] = None
        self.connected = False

    def connect(self) -> bool:
        """
        Connect to the server with retry logic and progress bar.

        Returns
        -------
        bool
            True if connection successful, False otherwise.
        """
        start_time = time.time()
        elapsed = 0

        if RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=None,
            ) as progress:
                task = progress.add_task(
                    "Waiting for server to start...",
                    total=self.timeout,
                )
                while elapsed < self.timeout:
                    try:
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        sock.settimeout(1)  # Short timeout for quick retries
                        result = sock.connect_ex((self.server_host, self.server_port))
                        if result == 0:
                            sock.settimeout(None)  # Remove timeout after connection
                            self.socket = sock
                            self.connected = True
                            progress.update(task, completed=self.timeout)
                            return True
                        sock.close()
                    except (socket.error, OSError):
                        pass

                    elapsed = time.time() - start_time
                    progress.update(task, completed=elapsed)
                    time.sleep(1)
        else:
            # Fallback to simple text progress
            print("Waiting for server to start...")
            while elapsed < self.timeout:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(1)
                    result = sock.connect_ex((self.server_host, self.server_port))
                    if result == 0:
                        sock.settimeout(None)
                        self.socket = sock
                        self.connected = True
                        print(f"\nConnected to server!")
                        return True
                    sock.close()
                except (socket.error, OSError):
                    pass

                elapsed = time.time() - start_time
                progress_pct = int((elapsed / self.timeout) * 100)
                remaining = int(self.timeout - elapsed)
                print(f"\rProgress: {progress_pct}% ({remaining}s remaining)", end="", flush=True)
                time.sleep(1)

        print(f"\nConnection timeout after {self.timeout} seconds")
        return False

    def send_message(self, message: Dict[str, Any]) -> None:
        """
        Send a message to the server.

        Parameters
        ----------
        message : Dict[str, Any]
            Message to send.

        Raises
        ------
        ConnectionError
            If not connected or connection lost.
        """
        if not self.connected or self.socket is None:
            raise ConnectionError("Not connected to server")

        try:
            data = serialize_message(message)
            # Send message length first (4 bytes)
            length = len(data)
            self.socket.sendall(length.to_bytes(4, byteorder="big"))
            # Send message data
            self.socket.sendall(data)
        except (socket.error, OSError) as e:
            self.connected = False
            raise ConnectionError(f"Failed to send message: {e}") from e

    def receive_message(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Receive a message from the server.

        Parameters
        ----------
        timeout : Optional[float]
            Optional timeout in seconds.

        Returns
        -------
        Optional[Dict[str, Any]]
            Received message, or None if timeout/error.

        Raises
        ------
        ConnectionError
            If connection lost.
        """
        if not self.connected or self.socket is None:
            raise ConnectionError("Not connected to server")

        try:
            if timeout is not None:
                self.socket.settimeout(timeout)
            else:
                self.socket.settimeout(None)

            # Receive message length (4 bytes)
            length_data = b""
            while len(length_data) < 4:
                chunk = self.socket.recv(4 - len(length_data))
                if not chunk:
                    self.connected = False
                    raise ConnectionError("Server closed connection")
                length_data += chunk

            length = int.from_bytes(length_data, byteorder="big")

            # Receive message data
            data = b""
            while len(data) < length:
                chunk = self.socket.recv(length - len(data))
                if not chunk:
                    self.connected = False
                    raise ConnectionError("Server closed connection")
                data += chunk

            return deserialize_message(data)
        except socket.timeout:
            return None
        except (socket.error, OSError) as e:
            self.connected = False
            raise ConnectionError(f"Failed to receive message: {e}") from e

    def send_decision(
        self,
        decision_type: DecisionType,
        decision: Any,
    ) -> None:
        """
        Send a decision response to the server.

        Parameters
        ----------
        decision_type : DecisionType
            Type of decision.
        decision : Any
            The decision value.
        """
        message = create_decision_response(decision_type, decision)
        self.send_message(message)

    def select_position(self, available_positions: list[int]) -> int:
        """
        Prompt user to select a position.

        Parameters
        ----------
        available_positions : list[int]
            List of available position IDs.

        Returns
        -------
        int
            Selected position ID.
        """
        print("\nAvailable positions:")
        position_names = {
            0: "Server (Player 0)",
            1: "Opponent (Player 1)",
            2: "Partner (Player 2)",
            3: "Opponent (Player 3)",
        }
        for pos_id in available_positions:
            print(f"  {pos_id}: {position_names.get(pos_id, f'Player {pos_id}')}")

        while True:
            try:
                choice = input(f"\nSelect position ({', '.join(map(str, available_positions))}): ").strip()
                pos_id = int(choice)
                if pos_id in available_positions:
                    return pos_id
                print(f"Invalid position. Please choose from {available_positions}")
            except ValueError:
                print("Invalid input. Please enter a number.")
            except (EOFError, KeyboardInterrupt):
                print("\nCancelled.")
                sys.exit(1)

    def close(self) -> None:
        """Close the connection to the server."""
        if self.socket is not None:
            try:
                self.socket.close()
            except Exception:
                pass
            self.socket = None
        self.connected = False

    def __enter__(self) -> "EuchreClient":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()

