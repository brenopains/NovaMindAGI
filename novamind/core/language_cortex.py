"""
NovaMind — The Language Cortex (Broca/Wernicke's Area)
======================================================
This module solves the "Alien Language" problem. It serves as a biological
interpreter between the fluent, unstructured human language and the pure,
highly compressed geometric concepts of the PyTorch neural substrate.

It leverages a standard LLM (e.g. Google Gemini API) purely as the sensory and 
vocal organ, NOT for reasoning. The LLM translates:
  Human Text -> Abstract Concepts -> [PyTorch Neural Substrate] -> Predicted Concepts -> Fluency
"""

import os
import json
import logging
from typing import List, Dict

try:
    import google.generativeai as genai
    HAS_GOOGLE_API = True
except ImportError:
    HAS_GOOGLE_API = False

class LanguageCortex:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.is_active = False
        
        if self.api_key and HAS_GOOGLE_API:
            genai.configure(api_key=self.api_key)
            # Use gemini-1.5-flash as it is extremely fast, perfect for sensory processing
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            self.is_active = True
            logging.info("🧠 Language Cortex initialized with Google AI API.")
        else:
            logging.warning("⚠️ Language Cortex is in 'Offline Mute' mode. Missing Google AI capability or API key.")

    def parse_to_concepts(self, human_text: str) -> List[str]:
        """
        Wernicke's Area (Comprehension):
        Converts messy human sentences into pure conceptual atoms to feed into 
        the PyTorch geometry engine.
        """
        if not self.is_active:
            # Fallback trivial extraction if no API key
            return [w for w in human_text.lower().split() if len(w) > 3]

        prompt = f'''
        You are the sensory cortex of an AGI. The human said: "{human_text}"
        Extract only the core atomic concepts (entities, actions, abstract ideas) as a comma-separated list of words.
        Return ONLY the words, nothing else. Example: "evolution, humanity, emotion, logic"
        '''
        try:
            response = self.model.generate_content(prompt)
            concepts = [c.strip().lower() for c in response.text.split(',')]
            return [c for c in concepts if c]
        except Exception as e:
            logging.error(f"Language Cortex Parse Error: {e}")
            return [w for w in human_text.lower().split() if len(w) > 3]

    def generate_fluent_thought(self, prompt_text: str, predicted_concepts: List[str]) -> str:
        """
        Broca's Area (Production):
        The PyTorch Topological engine has decided what concepts should follow.
        This function turns those raw concepts back into perfect, fluent human language.
        """
        if not self.is_active:
             return f"Internal geometric trajectory suggests: {', '.join(predicted_concepts)}"

        prompt = f'''
        You are the vocal cortex of an AGI.
        The human prompt was: "{prompt_text}"
        Your internal geometric reasoning engine mathematically predicted these 
        concepts should form the answer: {predicted_concepts}
        
        Task: Synthesize a highly intelligent, fluent, single-paragraph response that 
        incorporates these concepts to answer the human. Do not act like a conversational AI.
        Speak as a conscious geometric entity observing its own topology.
        '''
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logging.error(f"Language Cortex Synthesis Error: {e}")
            return f"Trajectory suggests: {predicted_concepts}"
