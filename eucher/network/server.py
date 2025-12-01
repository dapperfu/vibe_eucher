"""Server implementation for network multiplayer Euchre."""

import socket
import threading
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from eucher.network.protocol import (
    MessageType,
    DecisionType,
    create_message,
    create_decision_request,
    parse_decision_response,
    serialize_message,
    deserialize_message,
)


class ClientConnection:
    """Represents a connected client."""

    def __init__(self, socket: socket.socket, address: Tuple[str, int]) -> None:
        """
        Initialize client connection.

        Parameters
        ----------
        socket : socket.socket
            Client socket.
        address : Tuple[str, int]
            Client address (host, port).
        """
        self.socket = socket
        self.address = address
        self.player_id: Optional[int] = None
        self.name: Optional[str] = None
        self.connected = True
        self.pending_decision: Optional[Dict[str, Any]] = None
        self.decision_queue: deque = deque()
        self.lock = threading.Lock()

    def send_message(self, message: Dict[str, Any]) -> bool:
        """
        Send a message to the client.

        Parameters
        ----------
        message : Dict[str, Any]
            Message to send.

        Returns
        -------
        bool
            True if successful, False if connection lost.
        """
        if not self.connected:
            return False

        try:
            data = serialize_message(message)
            length = len(data)
            self.socket.sendall(length.to_bytes(4, byteorder="big"))
            self.socket.sendall(data)
            return True
        except (socket.error, OSError):
            self.connected = False
            return False

    def receive_message(self) -> Optional[Dict[str, Any]]:
        """
        Receive a message from the client.

        Returns
        -------
        Optional[Dict[str, Any]]
            Received message, or None if error/disconnect.
        """
        if not self.connected:
            return None

        try:
            # Receive message length
            length_data = b""
            while len(length_data) < 4:
                chunk = self.socket.recv(4 - len(length_data))
                if not chunk:
                    self.connected = False
                    return None
                length_data += chunk

            length = int.from_bytes(length_data, byteorder="big")

            # Receive message data
            data = b""
            while len(data) < length:
                chunk = self.socket.recv(length - len(data))
                if not chunk:
                    self.connected = False
                    return None
                data += chunk

            return deserialize_message(data)
        except (socket.error, OSError):
            self.connected = False
            return None

    def close(self) -> None:
        """Close the client connection."""
        self.connected = False
        try:
            self.socket.close()
        except Exception:
            pass


class EuchreServer:
    """Server for hosting network multiplayer Euchre games."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8765) -> None:
        """
        Initialize the server.

        Parameters
        ----------
        host : str
            Host address to bind to (default: "0.0.0.0" for all interfaces).
        port : int
            Port number to listen on (default: 8765).
        """
        self.host = host
        self.port = port
        self.socket: Optional[socket.socket] = None
        self.clients: List[ClientConnection] = []
        self.player_positions: Dict[int, Optional[int]] = {0: 0, 1: None, 2: None, 3: None}  # Server is position 0
        self.running = False
        self.lock = threading.Lock()
        self.decision_callbacks: Dict[int, Any] = {}  # player_id -> callback

    def start(self) -> None:
        """Start the server and begin accepting connections."""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host, self.port))
        self.socket.listen(4)  # Accept up to 4 connections (3 clients + buffer)
        self.socket.settimeout(1.0)  # Allow periodic checks
        self.running = True

        print(f"Server started on {self.host}:{self.port}")
        print("Waiting for clients to connect...")

        # Start accepting connections in a separate thread
        accept_thread = threading.Thread(target=self._accept_connections, daemon=True)
        accept_thread.start()

    def _accept_connections(self) -> None:
        """Accept incoming client connections."""
        while self.running:
            try:
                client_socket, address = self.socket.accept()
                print(f"Client connected from {address[0]}:{address[1]}")

                client = ClientConnection(client_socket, address)

                with self.lock:
                    self.clients.append(client)

                # Send connection acknowledgment
                ack_message = create_message(
                    MessageType.CONNECTION_ACK,
                    {"message": "Connected to Euchre server"},
                )
                client.send_message(ack_message)

                # Send available positions
                available = self.get_available_positions()
                position_message = create_message(
                    MessageType.POSITION_SELECT,
                    {"available_positions": available},
                )
                client.send_message(position_message)

                # Start client handler thread
                handler_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client,),
                    daemon=True,
                )
                handler_thread.start()
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"Error accepting connection: {e}")

    def _handle_client(self, client: ClientConnection) -> None:
        """Handle messages from a client."""
        while self.running and client.connected:
            message = client.receive_message()
            if message is None:
                break

            msg_type = MessageType(message["type"])

            if msg_type == MessageType.POSITION_SELECT:
                # Client selected a position
                position = message["data"].get("position")
                name = message["data"].get("name", f"Player {position}")
                if position is not None and self.assign_position(client, position, name):
                    # Notify all clients of new player
                    self.broadcast_message(
                        create_message(
                            MessageType.PLAYER_JOINED,
                            {
                                "player_id": position,
                                "name": name,
                            },
                        ),
                        exclude=client,
                    )
                    # Send confirmation to client
                    assigned_message = create_message(
                        MessageType.POSITION_ASSIGNED,
                        {
                            "player_id": position,
                            "name": name,
                        },
                        player_id=position,
                    )
                    client.send_message(assigned_message)
                else:
                    # Position assignment failed
                    error_message = create_message(
                        MessageType.ERROR,
                        {"message": "Position assignment failed"},
                    )
                    client.send_message(error_message)

            elif msg_type == MessageType.DECISION_RESPONSE:
                # Client sent a decision
                with client.lock:
                    decision_type = DecisionType(message["data"]["decision_type"])
                    decision = parse_decision_response(message)
                    client.decision_queue.append((decision_type, decision))

    def assign_position(self, client: ClientConnection, position: int, name: str) -> bool:
        """
        Assign a position to a client.

        Parameters
        ----------
        client : ClientConnection
            Client to assign position to.
        position : int
            Position ID (0-3).
        name : str
            Player name.

        Returns
        -------
        bool
            True if assignment successful, False otherwise.
        """
        with self.lock:
            if position < 0 or position > 3:
                return False
            if self.player_positions[position] is not None:
                return False  # Position already taken

            self.player_positions[position] = position
            client.player_id = position
            client.name = name
            return True

    def get_available_positions(self) -> List[int]:
        """
        Get list of available positions.

        Returns
        -------
        List[int]
            List of available position IDs.
        """
        with self.lock:
            return [
                pos_id
                for pos_id, assigned in self.player_positions.items()
                if assigned is None
            ]

    def get_client_by_player_id(self, player_id: int) -> Optional[ClientConnection]:
        """
        Get client connection for a player ID.

        Parameters
        ----------
        player_id : int
            Player ID.

        Returns
        -------
        Optional[ClientConnection]
            Client connection, or None if not found.
        """
        with self.lock:
            for client in self.clients:
                if client.player_id == player_id:
                    return client
        return None

    def broadcast_message(
        self,
        message: Dict[str, Any],
        exclude: Optional[ClientConnection] = None,
    ) -> None:
        """
        Broadcast a message to all connected clients.

        Parameters
        ----------
        message : Dict[str, Any]
            Message to broadcast.
        exclude : Optional[ClientConnection]
            Optional client to exclude from broadcast.
        """
        with self.lock:
            disconnected = []
            for client in self.clients:
                if client == exclude:
                    continue
                if not client.send_message(message):
                    disconnected.append(client)

            # Remove disconnected clients
            for client in disconnected:
                self.clients.remove(client)
                if client.player_id is not None:
                    self.player_positions[client.player_id] = None
                    # Notify other clients
                    disconnect_msg = create_message(
                        MessageType.PLAYER_DISCONNECTED,
                        {"player_id": client.player_id},
                    )
                    self.broadcast_message(disconnect_msg)

    def request_decision(
        self,
        player_id: int,
        decision_type: DecisionType,
        context: Dict[str, Any],
        timeout: Optional[float] = None,
    ) -> Optional[Any]:
        """
        Request a decision from a player (local or network).

        Parameters
        ----------
        player_id : int
            Player ID making the decision.
        decision_type : DecisionType
            Type of decision requested.
        context : Dict[str, Any]
            Context for the decision.
        timeout : Optional[float]
            Optional timeout in seconds.

        Returns
        -------
        Optional[Any]
            The decision, or None if timeout/error.
        """
        if player_id == 0:
            # Server player - return None to indicate local decision needed
            return None

        client = self.get_client_by_player_id(player_id)
        if client is None:
            return None

        # Send decision request
        request = create_decision_request(decision_type, player_id, context)
        if not client.send_message(request):
            return None

        # Wait for response
        import time
        start_time = time.time()
        while True:
            if timeout is not None and (time.time() - start_time) > timeout:
                return None

            with client.lock:
                if client.decision_queue:
                    decision_type_recv, decision = client.decision_queue.popleft()
                    if decision_type_recv == decision_type:
                        return decision

            time.sleep(0.1)  # Small delay to avoid busy waiting

    def is_player_connected(self, player_id: int) -> bool:
        """
        Check if a player is connected.

        Parameters
        ----------
        player_id : int
            Player ID to check.

        Returns
        -------
        bool
            True if player is connected.
        """
        if player_id == 0:
            return True  # Server player is always "connected"
        return self.get_client_by_player_id(player_id) is not None

    def get_connected_players(self) -> List[int]:
        """
        Get list of connected player IDs.

        Returns
        -------
        List[int]
            List of connected player IDs.
        """
        with self.lock:
            players = [0]  # Server player
            for client in self.clients:
                if client.player_id is not None:
                    players.append(client.player_id)
            return sorted(players)

    def stop(self) -> None:
        """Stop the server and close all connections."""
        self.running = False
        with self.lock:
            for client in self.clients:
                client.close()
            self.clients.clear()
        if self.socket is not None:
            try:
                self.socket.close()
            except Exception:
                pass

