"""
NovaMind — HTTP API Server
"""

import os
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from core.mind import NovaMind

# Initialize the Mind
mind = NovaMind()

# Flask app
app = Flask(__name__, static_folder='web', static_url_path='')
CORS(app)


@app.route('/')
def index():
    return send_from_directory('web', 'index.html')


@app.route('/api/think', methods=['POST'])
def think():
    """Submit input to the mind and get a complete thought trace."""
    data = request.get_json()
    raw_input = data.get('input', '')
    if not raw_input:
        return jsonify({'error': 'No input provided'}), 400

    thought = mind.think(raw_input)
    return jsonify(thought)


@app.route('/api/state', methods=['GET'])
def get_state():
    """Get the complete mind state."""
    state = mind.get_full_state()
    return jsonify(state)


@app.route('/api/graph', methods=['GET'])
def get_graph():
    """Get the knowledge graph data."""
    return jsonify(mind.world_model.get_graph_data())


@app.route('/api/memory', methods=['GET'])
def get_memory():
    """Get memory contents and stats."""
    return jsonify({
        'stats': mind.memory.get_stats(),
        'contents': mind.memory.get_all_memories(),
    })


@app.route('/api/metacognition', methods=['GET'])
def get_metacognition():
    """Get metacognitive state."""
    return jsonify(mind.metacognition.get_full_state())


@app.route('/api/goals', methods=['GET'])
def get_goals():
    """Get goal system state."""
    return jsonify(mind.goals.get_state())


@app.route('/api/learning', methods=['GET'])
def get_learning():
    """Get learning statistics."""
    return jsonify(mind.learning.get_stats())


@app.route('/api/concepts', methods=['GET'])
def get_concepts():
    """Get all known concepts."""
    return jsonify(mind.perception.get_all_concepts())


if __name__ == '__main__':
    print("\n" + "="*60)
    print("  🧠 NovaMind — 7-Layer AGI Architecture Prototype")
    print("="*60)
    print(f"  Dashboard: http://localhost:5000")
    print(f"  API:       http://localhost:5000/api/")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
