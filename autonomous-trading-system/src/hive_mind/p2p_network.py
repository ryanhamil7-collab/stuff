"""
P2P Network Module

Implements decentralized peer-to-peer network for hive mind collaboration.
Uses gRPC for communication and supports dynamic peer discovery.
"""

import asyncio
import hashlib
import json
import logging
import socket
import threading
import time
from concurrent import futures
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Optional, Set, Callable

import grpc
import yaml

logger = logging.getLogger(__name__)


@dataclass
class PeerInfo:
    """Information about a peer node"""
    peer_id: str
    host: str
    port: int
    last_seen: float
    reputation: float = 1.0
    total_contributions: int = 0
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'PeerInfo':
        return cls(**data)
    
    def is_alive(self, timeout: int = 300) -> bool:
        """Check if peer is still alive (seen within timeout)"""
        return (time.time() - self.last_seen) < timeout


class P2PNode:
    """
    P2P Node for Hive Mind Network
    
    Manages connections to peers, handles message routing,
    and coordinates collaborative learning.
    """
    
    def __init__(self, config: dict):
        """Initialize P2P node
        
        Args:
            config: Hive mind configuration dictionary
        """
        self.config = config
        self.node_id = self._generate_node_id()
        self.host = config['network']['host']
        self.port = config['network']['port']
        
        self.peers: Dict[str, PeerInfo] = {}
        self.peer_lock = threading.Lock()
        
        self.message_handlers: Dict[str, Callable] = {}
        
        self.server: Optional[grpc.Server] = None
        self.running = False
        
        self.stats = {
            'messages_sent': 0,
            'messages_received': 0,
            'updates_shared': 0,
            'updates_received': 0,
            'peers_connected': 0
        }
        
        logger.info(f"P2P Node initialized: {self.node_id}")
    
    def _generate_node_id(self) -> str:
        """Generate unique node ID based on hostname and timestamp"""
        hostname = socket.gethostname()
        timestamp = str(time.time())
        data = f"{hostname}:{timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def start(self):
        """Start P2P node and gRPC server"""
        logger.info(f"Starting P2P node on {self.host}:{self.port}")
        
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        self.server.add_insecure_port(f'{self.host}:{self.port}')
        self.server.start()
        
        self.running = True
        
        self._connect_to_peers()
        
        if self.config['network']['discovery_enabled']:
            threading.Thread(target=self._peer_discovery_loop, daemon=True).start()
        
        threading.Thread(target=self._heartbeat_loop, daemon=True).start()
        
        logger.info(f"P2P node started: {self.node_id}")
    
    def stop(self):
        """Stop P2P node"""
        logger.info("Stopping P2P node...")
        self.running = False
        
        if self.server:
            self.server.stop(grace=5)
        
        logger.info("P2P node stopped")
    
    def _connect_to_peers(self):
        """Connect to configured peers"""
        peer_addresses = self.config['network']['peers']
        
        for address in peer_addresses:
            try:
                host, port = address.split(':')
                self.add_peer(host, int(port))
            except Exception as e:
                logger.error(f"Failed to connect to peer {address}: {e}")
    
    def add_peer(self, host: str, port: int) -> bool:
        """Add a new peer to the network
        
        Args:
            host: Peer hostname/IP
            port: Peer port
            
        Returns:
            True if peer added successfully
        """
        try:
            peer_id = hashlib.sha256(f"{host}:{port}".encode()).hexdigest()[:16]
            
            if peer_id in self.peers:
                logger.debug(f"Already connected to peer {peer_id}")
                return True
            
            peer_info = PeerInfo(
                peer_id=peer_id,
                host=host,
                port=port,
                last_seen=time.time()
            )
            
            if self._test_peer_connection(host, port):
                with self.peer_lock:
                    self.peers[peer_id] = peer_info
                    self.stats['peers_connected'] += 1
                
                logger.info(f"Connected to peer {peer_id} ({host}:{port})")
                return True
            else:
                logger.warning(f"Failed to connect to peer {host}:{port}")
                return False
                
        except Exception as e:
            logger.error(f"Error adding peer {host}:{port}: {e}")
            return False
    
    def _test_peer_connection(self, host: str, port: int) -> bool:
        """Test if peer is reachable
        
        Args:
            host: Peer hostname/IP
            port: Peer port
            
        Returns:
            True if peer is reachable
        """
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except:
            return False
    
    def remove_peer(self, peer_id: str):
        """Remove a peer from the network
        
        Args:
            peer_id: ID of peer to remove
        """
        with self.peer_lock:
            if peer_id in self.peers:
                del self.peers[peer_id]
                logger.info(f"Removed peer {peer_id}")
    
    def get_active_peers(self) -> List[PeerInfo]:
        """Get list of active peers
        
        Returns:
            List of active peer info objects
        """
        with self.peer_lock:
            return [peer for peer in self.peers.values() if peer.is_alive()]
    
    def broadcast_message(self, message_type: str, data: dict):
        """Broadcast message to all active peers
        
        Args:
            message_type: Type of message
            data: Message data
        """
        active_peers = self.get_active_peers()
        
        for peer in active_peers:
            try:
                self.send_message(peer.peer_id, message_type, data)
            except Exception as e:
                logger.error(f"Failed to send message to {peer.peer_id}: {e}")
    
    def send_message(self, peer_id: str, message_type: str, data: dict):
        """Send message to specific peer
        
        Args:
            peer_id: Target peer ID
            message_type: Type of message
            data: Message data
        """
        if peer_id not in self.peers:
            logger.warning(f"Peer {peer_id} not found")
            return
        
        peer = self.peers[peer_id]
        
        message = {
            'from': self.node_id,
            'to': peer_id,
            'type': message_type,
            'timestamp': time.time(),
            'data': data
        }
        
        try:
            logger.debug(f"Sending {message_type} to {peer_id}")
            self.stats['messages_sent'] += 1
            
        except Exception as e:
            logger.error(f"Failed to send message to {peer_id}: {e}")
    
    def register_handler(self, message_type: str, handler: Callable):
        """Register handler for message type
        
        Args:
            message_type: Type of message to handle
            handler: Callback function
        """
        self.message_handlers[message_type] = handler
        logger.debug(f"Registered handler for {message_type}")
    
    def handle_message(self, message: dict):
        """Handle incoming message
        
        Args:
            message: Message dictionary
        """
        message_type = message.get('type')
        
        if message_type in self.message_handlers:
            try:
                self.message_handlers[message_type](message)
                self.stats['messages_received'] += 1
            except Exception as e:
                logger.error(f"Error handling {message_type}: {e}")
        else:
            logger.warning(f"No handler for message type: {message_type}")
    
    def _peer_discovery_loop(self):
        """Background loop for peer discovery"""
        logger.info("Starting peer discovery loop")
        
        while self.running:
            try:
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in peer discovery: {e}")
    
    def _heartbeat_loop(self):
        """Background loop for sending heartbeats"""
        logger.info("Starting heartbeat loop")
        
        while self.running:
            try:
                self.broadcast_message('heartbeat', {
                    'node_id': self.node_id,
                    'timestamp': time.time()
                })
                
                self._cleanup_dead_peers()
                
                time.sleep(30)  # Heartbeat every 30 seconds
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
    
    def _cleanup_dead_peers(self):
        """Remove peers that haven't been seen recently"""
        timeout = 300  # 5 minutes
        
        with self.peer_lock:
            dead_peers = [
                peer_id for peer_id, peer in self.peers.items()
                if not peer.is_alive(timeout)
            ]
            
            for peer_id in dead_peers:
                logger.info(f"Removing dead peer: {peer_id}")
                del self.peers[peer_id]
    
    def get_stats(self) -> dict:
        """Get node statistics
        
        Returns:
            Dictionary of statistics
        """
        return {
            **self.stats,
            'active_peers': len(self.get_active_peers()),
            'total_peers': len(self.peers)
        }


class P2PNetwork:
    """
    P2P Network Manager
    
    High-level interface for managing P2P network operations.
    """
    
    def __init__(self, config_path: str = "config/scheduler.yaml"):
        """Initialize P2P network
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.config = config['hive_mind']
        self.node: Optional[P2PNode] = None
        
        logger.info("P2P Network initialized")
    
    def start(self):
        """Start P2P network"""
        if not self.config['enabled']:
            logger.info("Hive mind disabled in config")
            return
        
        self.node = P2PNode(self.config)
        self.node.start()
        
        logger.info("P2P Network started")
    
    def stop(self):
        """Stop P2P network"""
        if self.node:
            self.node.stop()
        
        logger.info("P2P Network stopped")
    
    def is_running(self) -> bool:
        """Check if network is running
        
        Returns:
            True if network is running
        """
        return self.node is not None and self.node.running
    
    def get_node(self) -> Optional[P2PNode]:
        """Get P2P node instance
        
        Returns:
            P2PNode instance or None
        """
        return self.node
