"""
NovaMind — The Central Mind (Pipeline Orchestrator)
====================================================

This is the "consciousness" of NovaMind — the central process that
orchestrates all 7 cognitive layers into a unified processing pipeline.

For each input, the mind executes a COGNITIVE CYCLE:

    1. PERCEIVE  → Convert input to geometric concepts (Layer 1)
    2. MODEL     → Integrate into causal world model (Layer 2)
    3. REMEMBER  → Store in episodic memory, check compaction (Layer 3)
    4. REASON    → Hybrid reasoning (neural+symbolic+geometric) (Layer 4)
    5. REFLECT   → Metacognitive self-assessment (Layer 5)
    6. MOTIVATE  → Update goals and intrinsic drives (Layer 6)
    7. LEARN     → Adapt and improve continuously (Layer 7)

Each cycle produces a complete "thought trace" that shows exactly
what happened at every level — full cognitive transparency.
"""

import time
import json
from typing import Dict, Optional, List

from .perception import PerceptionEngine
from .world_model import WorldModel
from .memory import MemorySystem
from .reasoning import HybridReasoningEngine
from .metacognition import MetacognitionSystem
from .goals import GoalSystem
from .learning import ContinuousLearningEngine
from .language_cortex import NativeLanguageCortex

# NovaCrush Integration
try:
    from .novacrush.genetic_memory import GeneticOptimizer, Genome
    NOVACRUSH_AVAILABLE = True
except ImportError:
    NOVACRUSH_AVAILABLE = False


class NovaMind:
    """
    The central cognitive architecture — 7 layers unified.
    PURE AGI EVOLUTION: Zero LLM dependency.
    """

    def __init__(self):
        # Initialize all 7 layers (No LLM API)
        self.perception = PerceptionEngine()        # Layer 1
        self.world_model = WorldModel()             # Layer 2
        self.memory = MemorySystem()                # Layer 3
        self.reasoning = HybridReasoningEngine()    # Layer 4
        self.metacognition = MetacognitionSystem()  # Layer 5
        self.goals = GoalSystem()                   # Layer 6
        self.learning = ContinuousLearningEngine()  # Layer 7

        # Language Cortex (Broca's Area) - powered by NovaCrush natively
        self.language = NativeLanguageCortex(substrate=getattr(self.perception, 'neural_substrate', None))

        # NovaCrush: Genetic compression for periodic genome snapshots
        if NOVACRUSH_AVAILABLE:
            self.genetic_optimizer = GeneticOptimizer(population_size=10)
            self.genome_checkpoint_interval = 50  # Compress genome every N cycles
            self.last_genome = None
        else:
            self.genetic_optimizer = None

        # Cycle counter
        self.cycle_count = 0
        self.thought_history: List[Dict] = []
        self.start_time = time.time()

    def think(self, raw_input: str) -> Dict:
        cycle_start = time.time()
        self.cycle_count += 1

        thought = {
            'cycle': self.cycle_count,
            'input': raw_input,
            'timestamp': cycle_start,
            'layers': {},
        }

        # ═══════════════════════════════════════════════
        # LAYER 1: PURE GEOMETRIC PERCEPTION (No LLMs)
        # ═══════════════════════════════════════════════
        perception_report = self.perception.perceive(raw_input)
        thought['layers']['perception'] = perception_report

        # Decay old concept activations
        self.perception.decay_activations(rate=0.03)

        # ═══════════════════════════════════════════════
        # LAYER 2: WORLD MODEL
        # ═══════════════════════════════════════════════
        self.world_model.integrate_perception(perception_report)
        world_model_data = self.world_model.get_graph_data()
        thought['layers']['world_model'] = world_model_data

        # ═══════════════════════════════════════════════
        # LAYER 3: MEMORY
        # ═══════════════════════════════════════════════
        episode = self.memory.store_episode(
            content={
                'input': raw_input,
                'concepts': [c['label'] for c in perception_report.get('concepts', [])],
                'relations': len(perception_report.get('relations', [])),
                'type': 'interaction',
            },
            episode_type='interaction',
            importance=0.5 + 0.3 * (perception_report.get('new_concepts', 0) > 0)
        )

        relevant_episodes = self.memory.recall_episodic({'input': raw_input}, top_k=3)
        relevant_rules = self.memory.recall_semantic({'input': raw_input}, top_k=3)

        memory_report = {
            'stored_episode': episode.id,
            'recalled_episodes': [ep.to_dict() for ep in relevant_episodes],
            'recalled_rules': [r.to_dict() for r in relevant_rules],
            'stats': self.memory.get_stats(),
        }
        thought['layers']['memory'] = memory_report
        self.memory.decay_all()

        # ═══════════════════════════════════════════════
        # LAYER 4: REASONING 
        # ═══════════════════════════════════════════════
        knowledge_base = self._build_knowledge_base(
            perception_report, world_model_data, relevant_rules
        )
        reasoning_report = self.reasoning.reason(
            perception_report.get('concepts', []),
            knowledge_base
        )
        thought['layers']['reasoning'] = reasoning_report

        # ═══════════════════════════════════════════════
        # LAYER 5: METACOGNITION
        # ═══════════════════════════════════════════════
        metacognition_report = self.metacognition.assess(
            perception_report, reasoning_report, memory_report['stats'], world_model_data
        )
        thought['layers']['metacognition'] = metacognition_report

        # ═══════════════════════════════════════════════
        # LAYER 6: GOALS
        # ═══════════════════════════════════════════════
        self.goals.update_from_metacognition(metacognition_report)
        current_goal = self.goals.get_next_goal()
        if current_goal:
            current_goal.update_progress(0.1, f"Processed cycle {self.cycle_count}")
        goal_report = self.goals.get_state()
        thought['layers']['goals'] = goal_report

        # ═══════════════════════════════════════════════
        # LAYER 7: LEARNING
        # ═══════════════════════════════════════════════
        learning_report = self.learning.learn(
            perception_report, reasoning_report, metacognition_report, memory_report['stats']
        )
        thought['layers']['learning'] = learning_report

        # NovaCrush: Periodic genetic compression checkpoint (saves genome snapshot)
        if (self.genetic_optimizer and 
            self.cycle_count % self.genome_checkpoint_interval == 0 and
            self.cycle_count > 0):
            try:
                import torch
                substrate = self.perception.neural_substrate
                weights = {
                    'embeddings': substrate.embeddings.data.clone(),
                }
                if hasattr(substrate, 'transition'):
                    weights['transition'] = substrate.transition.weight.data.clone()
                self.last_genome = self.genetic_optimizer.compress_weights(
                    weights, generations=10
                )
            except Exception:
                pass  # Non-critical — genome checkpoint is optional

        # ═══════════════════════════════════════════════
        # SYNTHESIS (Generating Native Signal Array)
        # ═══════════════════════════════════════════════
        response = self._synthesize_response(thought)
        thought['response'] = response
        thought['cycle_time_ms'] = (time.time() - cycle_start) * 1000

        self.thought_history.append({
            'cycle': self.cycle_count,
            'input': raw_input[:100],
            'response_preview': response['text'][:200],
            'confidence': metacognition_report.get('confidence', {}).get('overall', 0),
            'cycle_time_ms': thought['cycle_time_ms'],
        })
        if len(self.thought_history) > 50:
            self.thought_history = self.thought_history[-50:]

        return thought

    def _build_knowledge_base(self, perception: Dict, world_model: Dict,
                               semantic_rules: List) -> Dict:
        return {
            'concepts': self.perception.get_all_concepts(),
            'causal_edges': world_model.get('edges', []),
            'causal_chains': self.world_model.get_causal_chains(),
            'semantic_rules': [r.to_dict() for r in semantic_rules],
        }

    def _synthesize_response(self, thought: Dict) -> Dict:
        """
        Pure PyTorch Geometric Translation 
        No LLM String Manipulation - Just Pure Native Signal Vectoring
        """
        reasoning = thought['layers'].get('reasoning', {})
        metacognition = thought['layers'].get('metacognition', {})
        learning = thought['layers'].get('learning', {})
        perception_data = thought['layers'].get('perception', {})

        consensus = reasoning.get('consensus', {})
        reasoning_conclusion = consensus.get('conclusion', 'Processing...')
        
        # Native Language Generation!
        # Convert conclusion string to concepts
        predicted_concepts = self.language.parse_to_concepts(reasoning_conclusion)
        # Generate fluent sentence
        fluent_response = self.language.generate_fluent_thought(thought.get('raw_input', ''), predicted_concepts)
        
        confidence = metacognition.get('confidence', {}).get('overall', 0.5)

        parts = []
        parts.append(f"**NovaCrush Native Response:** {fluent_response}")
        parts.append(f"*(Topological Prediction: {reasoning_conclusion})*")

        # Self-assessment
        self_assessment = metacognition.get('self_assessment', {})
        assessment_text = self_assessment.get('text', '')
        if assessment_text:
            parts.append(f"\n**Self-Assessment:** {assessment_text}")

        # Learning highlights
        novelty = learning.get('novelty_analysis', {})
        if novelty.get('is_novel'):
            parts.append(f"\n**NEW: Novel input detected** -- updating knowledge base")

        # Knowledge metrics
        concept_count = perception_data.get('total_concepts_known', 0)

        # NovaCrush compression indicator
        crush_info = ''
        if hasattr(self.perception, 'novacrush_enabled') and self.perception.novacrush_enabled:
            crush_info = ' | NovaCrush ON'

        parts.append(f"\n*[Cycle #{thought['cycle']} | "
                    f"{concept_count} concepts | "
                    f"{confidence:.0%} confident | "
                    f"{thought.get('cycle_time_ms', 0):.0f}ms{crush_info}]*")

        return {
            'text': '\n'.join(parts),
            'confidence': confidence,
            'primary_paradigm': consensus.get('primary_paradigm', 'unknown'),
            'novel_input': novelty.get('is_novel', False),
        }

    def get_full_state(self) -> Dict:
        """Return the complete mind state for the dashboard."""
        state = {
            'cycle_count': self.cycle_count,
            'uptime_seconds': time.time() - self.start_time,
            'perception': {
                'total_concepts': len(self.perception.concept_registry),
                'all_concepts': self.perception.get_all_concepts(),
                'novacrush_enabled': getattr(self.perception, 'novacrush_enabled', False),
            },
            'world_model': self.world_model.get_graph_data(),
            'memory': {
                'stats': self.memory.get_stats(),
                'contents': self.memory.get_all_memories(),
            },
            'reasoning': {
                'paradigm_weights': self.reasoning.paradigm_weights,
                'history': self.reasoning.reasoning_history[-10:],
            },
            'metacognition': self.metacognition.get_full_state(),
            'goals': self.goals.get_state(),
            'learning': self.learning.get_stats(),
            'thought_history': self.thought_history,
        }

        # NovaCrush stats
        if NOVACRUSH_AVAILABLE and hasattr(self.perception, 'novacrush_enabled'):
            if self.perception.novacrush_enabled:
                substrate = self.perception.neural_substrate
                state['novacrush'] = {
                    'compression': substrate.get_compression_stats(),
                    'forward_forward': substrate.get_ff_stats(),
                    'last_genome_bytes': len(self.last_genome.serialize()) if self.last_genome else None,
                }

        return state
