"""
Hive Mind P2P Module

Decentralized collaborative learning system for autonomous trading.
Enables multiple users' instances to share and learn from novel discoveries
without compromising privacy.

Features:
- P2P gossip protocol for model updates
- Privacy-preserving gradient sharing
- Anti-poisoning with blockchain verification
- Incentive system for contributions
- Federated learning across nodes
"""

from .p2p_network import P2PNode, P2PNetwork
from .gossip_protocol import GossipProtocol
from .model_sharing import ModelSharer, SharedUpdate
from .anti_poisoning import AntiPoisoningValidator
from .incentive_system import IncentiveManager

__all__ = [
    'P2PNode',
    'P2PNetwork',
    'GossipProtocol',
    'ModelSharer',
    'SharedUpdate',
    'AntiPoisoningValidator',
    'IncentiveManager'
]
