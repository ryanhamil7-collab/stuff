"""
Gossip Protocol Module

Implements gossip-based message dissemination for P2P network.
Supports push, pull, and push-pull gossip strategies.
"""

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class GossipMessage:
    """Gossip message structure"""
    message_id: str
    message_type: str
    payload: dict
    sender_id: str
    timestamp: float
    ttl: int = 5
    hops: int = 0
    seen_by: Set[str] = field(default_factory=set)
    
    def is_expired(self, max_age: int = 3600) -> bool:
        """Check if message has expired"""
        return (time.time() - self.timestamp) > max_age
    
    def should_forward(self, max_hops: int = 5) -> bool:
        """Check if message should be forwarded"""
        return self.hops < max_hops and self.ttl > 0


class GossipProtocol:
    """
    Gossip Protocol Implementation
    
    Implements epidemic-style message dissemination across P2P network.
    """
    
    def __init__(self, node, config: dict):
        """Initialize gossip protocol
        
        Args:
            node: P2PNode instance
            config: Gossip configuration
        """
        self.node = node
        self.config = config
        
        self.message_cache: Dict[str, GossipMessage] = {}
        self.cache_size_limit = 1000
        
        self.stats = {
            'messages_gossiped': 0,
            'messages_received': 0,
            'messages_forwarded': 0,
            'messages_dropped': 0
        }
        
        logger.info("Gossip protocol initialized")
    
    def gossip(self, message_type: str, payload: dict):
        """Initiate gossip of a message
        
        Args:
            message_type: Type of message
            payload: Message payload
        """
        message = GossipMessage(
            message_id=str(uuid4()),
            message_type=message_type,
            payload=payload,
            sender_id=self.node.node_id,
            timestamp=time.time(),
            ttl=self.config.get('max_hops', 5)
        )
        
        self.message_cache[message.message_id] = message
        
        self._forward_message(message)
        
        self.stats['messages_gossiped'] += 1
        logger.debug(f"Gossiped message {message.message_id}")
    
    def receive_gossip(self, message_data: dict):
        """Receive and process gossip message
        
        Args:
            message_data: Message data dictionary
        """
        message_id = message_data.get('message_id')
        
        if message_id in self.message_cache:
            logger.debug(f"Already seen message {message_id}")
            return
        
        message = GossipMessage(**message_data)
        
        if message.is_expired():
            self.stats['messages_dropped'] += 1
            return
        
        self.message_cache[message_id] = message
        self._cleanup_cache()
        
        message.seen_by.add(self.node.node_id)
        message.hops += 1
        message.ttl -= 1
        
        self.stats['messages_received'] += 1
        
        self._process_message(message)
        
        if message.should_forward(self.config.get('max_hops', 5)):
            self._forward_message(message)
    
    def _forward_message(self, message: GossipMessage):
        """Forward message to random peers
        
        Args:
            message: Message to forward
        """
        active_peers = self.node.get_active_peers()
        
        unseen_peers = [
            peer for peer in active_peers
            if peer.peer_id not in message.seen_by
        ]
        
        if not unseen_peers:
            return
        
        fanout = min(self.config.get('fanout', 3), len(unseen_peers))
        selected_peers = random.sample(unseen_peers, fanout)
        
        for peer in selected_peers:
            try:
                message.seen_by.add(peer.peer_id)
                self.node.send_message(
                    peer.peer_id,
                    'gossip',
                    message.__dict__
                )
                self.stats['messages_forwarded'] += 1
            except Exception as e:
                logger.error(f"Failed to forward to {peer.peer_id}: {e}")
    
    def _process_message(self, message: GossipMessage):
        """Process received gossip message
        
        Args:
            message: Message to process
        """
        handler_name = f"_handle_{message.message_type}"
        
        if hasattr(self, handler_name):
            handler = getattr(self, handler_name)
            try:
                handler(message)
            except Exception as e:
                logger.error(f"Error processing {message.message_type}: {e}")
        else:
            logger.warning(f"No handler for message type: {message.message_type}")
    
    def _cleanup_cache(self):
        """Clean up message cache"""
        if len(self.message_cache) > self.cache_size_limit:
            sorted_messages = sorted(
                self.message_cache.items(),
                key=lambda x: x[1].timestamp
            )
            
            self.message_cache = dict(sorted_messages[-self.cache_size_limit:])
    
    def get_stats(self) -> dict:
        """Get gossip statistics
        
        Returns:
            Dictionary of statistics
        """
        return {
            **self.stats,
            'cache_size': len(self.message_cache)
        }
