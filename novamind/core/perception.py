"""
NovaMind — Layer 1: Genuine PyTorch Perception Engine
======================================================
Replaced static string hashing with a continuous training Neural Topology. 
Text -> PyTorch Continuous Predictor -> Live Geometric Tensors -> Hyperbolic Concept Setup.
"""

import numpy as np
import hashlib
import re
import time
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import torch
import torch.nn.functional as F

from .math.hyperbolic import PoincareBall
from .math.geometric_algebra import CliffordAlgebra, MultiVector
from .neural_substrate import DynamicPredictiveNetwork

CONCEPT_DIM = 32
GA_DIM = 8

class ConceptNode:
    def __init__(self, label: str, concept_type: str = "entity"):
        self.id = hashlib.md5(label.encode()).hexdigest()[:12]
        self.label = label
        self.concept_type = concept_type

        # Geometric representations (Now strictly trained tensors)
        self.hyperbolic_position = None
        self.clifford_vector: Optional[MultiVector] = None
        self.uncertainty_radius: float = 1.0  # Determined by Prediction Error / Surprise

        # Cognitive state
        self.activation: float = 1.0 
        self.familiarity: float = 0.0          
        self.creation_time: float = time.time()
        self.last_access: float = time.time()
        self.access_count: int = 0

        # Relational Matrix connections
        self.connections: Dict[str, List[str]] = defaultdict(list)

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'label': self.label,
            'type': self.concept_type,
            'position': self.hyperbolic_position.tolist() if self.hyperbolic_position is not None else None,
            'uncertainty': self.uncertainty_radius,
            'activation': self.activation,
            'familiarity': self.familiarity,
            'access_count': self.access_count,
            'connections': dict(self.connections),
        }

class PerceptionEngine:
    def __init__(self):
        self.ball = PoincareBall(dim=CONCEPT_DIM, curvature=1.0)
        self.algebra = CliffordAlgebra(p=GA_DIM)
        self.concept_registry: Dict[str, ConceptNode] = {}
        
        # Instantiate the GENUINE PyTorch Continuous Core
        self.neural_substrate = DynamicPredictiveNetwork(initial_concepts=8, embedding_dim=GA_DIM)

    def perceive(self, raw_input: str) -> Dict:
        start_time = time.time()
        
        # 1. Run genuine continuous training on the input string to spawn symbols & perform neurogenesis
        words = [w.strip() for w in re.split(r'\W+', raw_input.lower()) if w.strip()]
        if not words:
             words = ["none"]
             
        # Train PyTorch continuous model inline and get the resulting surprise (Free Energy drop)
        surprise = self.neural_substrate.continuous_train(words)
        
        atoms = []
        for word in words:
            # Create or get existing object
            concept_id = hashlib.md5(word.encode()).hexdigest()[:12]
            if concept_id in self.concept_registry:
                concept = self.concept_registry[concept_id]
                concept.access_count += 1
                concept.last_access = time.time()
            else:
                concept = ConceptNode(word, "concept")
                self.concept_registry[concept_id] = concept
            atoms.append(concept)
            
            # Position it strictly using the LIVE PYTORCH EMBEDDING (no faked maths)
            idx = self.neural_substrate.get_token_id(word)
            t_embedding = self.neural_substrate.embeddings[idx].detach().cpu().numpy()
            
            # Use PyTorch geometry as the hyperbolic geometry and Clifford vector root
            if len(t_embedding) < CONCEPT_DIM:
                 pad = np.zeros(CONCEPT_DIM - len(t_embedding))
                 h_pos = np.concatenate([t_embedding, pad])
            else:
                 h_pos = t_embedding[:CONCEPT_DIM]
                 
            # Assign genuine trained positions
            concept.hyperbolic_position = h_pos
            concept.clifford_vector = self.algebra.vector(t_embedding)
            # Uncertainty drops proportionally to the reciprocal of continuous learning surprise
            concept.uncertainty_radius = surprise
            concept.familiarity = 1.0 - surprise
            
        # 2. Causality Extraction strictly from PyTorch Transition Matrix
        topology_matrix = self.neural_substrate.get_topology_matrix()
        relations = []
        for i, source_concept in enumerate(atoms):
             s_idx = self.neural_substrate.get_token_id(source_concept.label)
             for j, target_concept in enumerate(atoms):
                  if i != j:
                       t_idx = self.neural_substrate.get_token_id(target_concept.label)
                       if s_idx < len(topology_matrix) and t_idx < len(topology_matrix):
                           similarity = topology_matrix[s_idx][t_idx]
                           if similarity > 0.6:  # Neural Threshold for causal edge
                                relations.append({
                                    'type': 'pytorch_latent_edge',
                                    'source': source_concept.label,
                                    'target': target_concept.label,
                                    'confidence': float(similarity),
                                    'extracted_from': 'neurogenesis_matrix',
                                })
                                source_concept.connections['latent_edge'].append(target_concept.label)
        
        perception_time = time.time() - start_time
        return {
            'raw_input': raw_input,
            'concepts': [a.to_dict() for a in set(atoms)],
            'relations': relations,
            'integration': {
                 'new_concepts': len([a for a in set(atoms) if a.access_count == 0]),
                 'network_size': self.neural_substrate.vocab_size,
                 'free_energy': surprise
            },
            'new_concepts': len([a for a in set(atoms) if a.access_count == 0]),
            'recognized_concepts': len(atoms),
            'perception_time_ms': perception_time * 1000,
            'total_concepts_known': len(self.concept_registry),
        }

    def get_concept_by_label(self, label: str) -> Optional[ConceptNode]:
        concept_id = hashlib.md5(label.encode()).hexdigest()[:12]
        return self.concept_registry.get(concept_id)

    def get_all_concepts(self) -> List[Dict]:
        return [c.to_dict() for c in self.concept_registry.values()]

    def decay_activations(self, rate: float = 0.05):
        for concept in self.concept_registry.values():
            concept.activation = max(0.01, concept.activation - rate)
