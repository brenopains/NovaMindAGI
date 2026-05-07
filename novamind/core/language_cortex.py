"""
NovaMind — The Native Language Cortex (Broca/Wernicke's Area)
=============================================================
This module is the biological interpreter between human language and 
the pure geometric concepts of the PyTorch neural substrate.

DeepSeek's "Andaime Semântico" (Semantic Skeleton) Architecture:
1. Wernicke's Area (Comprehension): Maps input words to geometric concepts.
2. Broca's Area (Production): Translates geometric predictions back to human text
   using the NativeLanguageGenerator (CrushedSubstrate transitions).
   
Zero external LLM. Pure continuous learning.
"""

import logging
from typing import List, Dict

try:
    from .novacrush.language_gen import NativeLanguageGenerator
    from .novacrush.crushed_substrate import CrushedSubstrate
    NOVACRUSH_ENABLED = True
except ImportError:
    NOVACRUSH_ENABLED = False

class NativeLanguageCortex:
    def __init__(self, substrate=None):
        self.is_active = True
        self.substrate = substrate
        self.generator = None
        
        if NOVACRUSH_ENABLED and self.substrate:
            self.generator = NativeLanguageGenerator(self.substrate)
            logging.info("🧠 Native Language Cortex initialized with NovaCrush Engine.")
        else:
            logging.warning("⚠️ Native Language Cortex is active but missing NovaCrush substrate.")

    def parse_to_concepts(self, human_text: str) -> List[str]:
        """
        Wernicke's Area (Comprehension):
        Converts messy human sentences into pure conceptual atoms.
        In the native version, we extract base tokens that the substrate knows.
        """
        # Simple tokenization
        words = [w.strip().lower() for w in human_text.replace('.', ' ').replace(',', ' ').split() if len(w) > 2]
        return words

    def generate_fluent_thought(self, prompt_text: str, predicted_concepts: List[str]) -> str:
        """
        Broca's Area (Production):
        The PyTorch Topological engine has decided what concepts should follow.
        This uses the learned transitions to generate text.
        """
        if not self.generator:
            return f"Internal trajectory: {', '.join(predicted_concepts)}"

        if not predicted_concepts:
            return "..."

        # Seed generation with the most critical predicted concept
        seed_concept = predicted_concepts[0]
        
        # Generate the sentence by walking the substrate's causal transition matrix
        result = self.generator.generate(
            seed=seed_concept, 
            max_tokens=15, 
            temperature=0.3,
            stop_on_repeat=True
        )
        
        text = result['text']
        
        # Capitalize first letter and add period
        if text:
            text = text[0].upper() + text[1:] + "."
            
        return text
