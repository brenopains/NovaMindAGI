/**
 * NovaMind Dashboard — Frontend Logic
 * ====================================
 * Real-time visualization of the 7-layer cognitive architecture.
 * D3.js for knowledge graph, SVG gauges for metacognition,
 * animated pipeline, and interactive chat.
 */

const API = '';

// ═══════════════════════════════════════
// STATE
// ═══════════════════════════════════════
let state = {
    cycles: 0,
    thinking: false,
    graphData: { nodes: [], edges: [] },
    lastThought: null,
};

// ═══════════════════════════════════════
// CHAT
// ═══════════════════════════════════════
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const chatMessages = document.getElementById('chat-messages');

chatInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

sendBtn.addEventListener('click', sendMessage);

async function sendMessage() {
    const text = chatInput.value.trim();
    if (!text || state.thinking) return;

    // Add user message
    addChatMessage(text, 'user');
    chatInput.value = '';

    // Show thinking state
    state.thinking = true;
    sendBtn.disabled = true;
    sendBtn.textContent = '...';
    animatePipeline();

    const loadingId = addLoadingMessage();

    try {
        const response = await fetch(`${API}/api/think`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ input: text }),
        });

        const thought = await response.json();
        state.lastThought = thought;
        state.cycles = thought.cycle || state.cycles + 1;

        // Remove loading
        removeMessage(loadingId);

        // Add response
        const responseText = thought.response?.text || 'Processing complete.';
        addChatMessage(responseText, 'system');

        // Update all panels
        updatePipeline(thought);
        updateGraph(thought.layers?.world_model);
        updateMemory(thought.layers?.memory);
        updateReasoningTrace(thought.layers?.reasoning);
        updateMetacognition(thought.layers?.metacognition);
        updateGoals(thought.layers?.goals);
        updateHeaderStats(thought);

    } catch (err) {
        removeMessage(loadingId);
        addChatMessage(`⚠️ Connection error: ${err.message}. Is the server running on port 5000?`, 'system');
    }

    state.thinking = false;
    sendBtn.disabled = false;
    sendBtn.textContent = 'Think';
    stopPipelineAnimation();
}

function addChatMessage(text, type) {
    const div = document.createElement('div');
    div.className = `message message-${type}`;
    // Parse basic markdown
    let html = text
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/\n/g, '<br>');
    div.innerHTML = html;
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return div;
}

function addLoadingMessage() {
    const div = document.createElement('div');
    div.className = 'loading';
    div.id = 'loading-msg';
    div.innerHTML = '<div class="spinner"></div> Processing through 7 cognitive layers...';
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return 'loading-msg';
}

function removeMessage(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

// ═══════════════════════════════════════
// PIPELINE ANIMATION
// ═══════════════════════════════════════
let pipelineInterval = null;
let activeLayerIndex = 0;
const layerIds = ['perception', 'world_model', 'memory', 'reasoning', 'metacognition', 'goals', 'learning'];

function animatePipeline() {
    activeLayerIndex = 0;
    pipelineInterval = setInterval(() => {
        layerIds.forEach(id => {
            document.getElementById(`layer-${id}`)?.classList.remove('active');
        });
        if (activeLayerIndex < layerIds.length) {
            document.getElementById(`layer-${layerIds[activeLayerIndex]}`)?.classList.add('active');
            activeLayerIndex++;
        } else {
            activeLayerIndex = 0;
        }
    }, 300);
}

function stopPipelineAnimation() {
    clearInterval(pipelineInterval);
    layerIds.forEach(id => {
        document.getElementById(`layer-${id}`)?.classList.remove('active');
    });
}

function updatePipeline(thought) {
    const layers = thought.layers || {};

    // Perception metrics
    const perception = layers.perception || {};
    document.getElementById('metric-perception').textContent =
        `${perception.new_concepts || 0} new`;

    // World model
    const wm = layers.world_model || {};
    document.getElementById('metric-world_model').textContent =
        `${wm.node_count || 0} nodes`;

    // Memory
    const mem = layers.memory?.stats?.episodic || {};
    document.getElementById('metric-memory').textContent =
        `${mem.currently_active || 0} active`;

    // Reasoning
    const reasoning = layers.reasoning?.consensus || {};
    document.getElementById('metric-reasoning').textContent =
        `${(reasoning.confidence * 100 || 0).toFixed(0)}%`;

    // Metacognition
    const meta = layers.metacognition?.confidence || {};
    document.getElementById('metric-metacognition').textContent =
        `${((meta.overall || 0) * 100).toFixed(0)}%`;

    // Goals
    const goals = layers.goals || {};
    document.getElementById('metric-goals').textContent =
        `${(goals.active_goals || []).length} active`;

    // Learning
    const learning = layers.learning || {};
    document.getElementById('metric-learning').textContent =
        `lr=${(learning.learning_rate || 0).toFixed(3)}`;
}

// ═══════════════════════════════════════
// KNOWLEDGE GRAPH (D3.js)
// ═══════════════════════════════════════
const graphSvg = d3.select('#graph-canvas');
let graphSimulation = null;

function initGraph() {
    const container = document.getElementById('graph-canvas');
    const rect = container.getBoundingClientRect();
    graphSvg.attr('width', rect.width).attr('height', 250);
}

function updateGraph(worldModel) {
    if (!worldModel) return;

    const container = document.getElementById('graph-canvas');
    const rect = container.getBoundingClientRect();
    const width = rect.width || 400;
    const height = 250;

    graphSvg.attr('width', width).attr('height', height);

    const nodes = (worldModel.nodes || []).map(n => ({
        id: n.id,
        label: n.label,
        type: n.type,
    }));

    const nodeIds = new Set(nodes.map(n => n.id));
    const links = (worldModel.edges || [])
        .filter(e => {
            const src = nodes.find(n => n.label === e.source || n.id === e.source);
            const tgt = nodes.find(n => n.label === e.target || n.id === e.target);
            return src && tgt;
        })
        .map(e => {
            const src = nodes.find(n => n.label === e.source || n.id === e.source);
            const tgt = nodes.find(n => n.label === e.target || n.id === e.target);
            return {
                source: src.id,
                target: tgt.id,
                type: e.type,
                strength: e.strength || 0.5,
            };
        });

    // Clear
    graphSvg.selectAll('*').remove();

    if (nodes.length === 0) {
        graphSvg.append('text')
            .attr('x', width / 2).attr('y', height / 2)
            .attr('text-anchor', 'middle')
            .attr('fill', '#555570')
            .attr('font-size', '12px')
            .text('Knowledge graph will appear here...');
        return;
    }

    // Defs for arrow markers
    graphSvg.append('defs').append('marker')
        .attr('id', 'arrowhead')
        .attr('viewBox', '0 -5 10 10')
        .attr('refX', 20).attr('refY', 0)
        .attr('markerWidth', 6).attr('markerHeight', 6)
        .attr('orient', 'auto')
        .append('path')
        .attr('d', 'M0,-5L10,0L0,5')
        .attr('fill', 'rgba(0,245,255,0.3)');

    // Simulation
    if (graphSimulation) graphSimulation.stop();

    graphSimulation = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(links).id(d => d.id).distance(60))
        .force('charge', d3.forceManyBody().strength(-120))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide().radius(25));

    // Links
    const link = graphSvg.append('g')
        .selectAll('line')
        .data(links)
        .enter().append('line')
        .attr('stroke', d => {
            const colors = {
                causes: 'rgba(0,245,255,0.4)',
                is_a: 'rgba(168,85,247,0.4)',
                has: 'rgba(34,197,94,0.4)',
                correlates: 'rgba(255,255,255,0.1)',
            };
            return colors[d.type] || 'rgba(255,255,255,0.15)';
        })
        .attr('stroke-width', d => Math.max(1, d.strength * 3))
        .attr('marker-end', 'url(#arrowhead)');

    // Nodes
    const node = graphSvg.append('g')
        .selectAll('circle')
        .data(nodes)
        .enter().append('circle')
        .attr('r', 6)
        .attr('fill', d => {
            const colors = {
                entity: '#00f5ff',
                action: '#a855f7',
                property: '#22c55e',
                abstract: '#fbbf24',
                relation: '#f43f5e',
                variable: '#3b82f6',
            };
            return colors[d.type] || '#888';
        })
        .attr('stroke', 'rgba(255,255,255,0.2)')
        .attr('stroke-width', 1)
        .call(d3.drag()
            .on('start', dragstarted)
            .on('drag', dragged)
            .on('end', dragended));

    // Labels
    const labels = graphSvg.append('g')
        .selectAll('text')
        .data(nodes)
        .enter().append('text')
        .attr('class', 'concept-label')
        .attr('dy', -10)
        .attr('text-anchor', 'middle')
        .text(d => d.label.length > 15 ? d.label.substring(0, 15) + '…' : d.label);

    graphSimulation.on('tick', () => {
        link
            .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
        node
            .attr('cx', d => d.x = Math.max(10, Math.min(width - 10, d.x)))
            .attr('cy', d => d.y = Math.max(10, Math.min(height - 10, d.y)));
        labels
            .attr('x', d => d.x).attr('y', d => d.y);
    });

    // Update badge
    document.getElementById('badge-nodes').textContent = `${nodes.length} nodes`;

    function dragstarted(event, d) {
        if (!event.active) graphSimulation.alphaTarget(0.3).restart();
        d.fx = d.x; d.fy = d.y;
    }
    function dragged(event, d) {
        d.fx = event.x; d.fy = event.y;
    }
    function dragended(event, d) {
        if (!event.active) graphSimulation.alphaTarget(0);
        d.fx = null; d.fy = null;
    }
}

// ═══════════════════════════════════════
// MEMORY PANEL
// ═══════════════════════════════════════
function updateMemory(memoryData) {
    if (!memoryData) return;

    const stats = memoryData.stats || {};
    const episodic = stats.episodic || {};
    const semantic = stats.semantic || {};
    const procedural = stats.procedural || {};

    document.getElementById('mem-episodic').textContent = episodic.currently_active || 0;
    document.getElementById('mem-semantic').textContent = semantic.total_rules || 0;
    document.getElementById('mem-procedural').textContent = procedural.total_procedures || 0;

    const compressionStats = semantic.total_compression || {};
    const ratio = (compressionStats.overall_ratio || 0) * 100;
    document.getElementById('compression-fill').style.width = `${ratio}%`;
    document.getElementById('compression-label').textContent = `${ratio.toFixed(1)}%`;
    document.getElementById('badge-compression').textContent = `${ratio.toFixed(0)}% compressed`;
}

// ═══════════════════════════════════════
// REASONING TRACE
// ═══════════════════════════════════════
function updateReasoningTrace(reasoning) {
    if (!reasoning) return;

    const traceList = document.getElementById('trace-list');
    traceList.innerHTML = '';

    const trace = reasoning.combined_trace || [];
    trace.forEach(line => {
        const div = document.createElement('div');
        div.className = 'trace-line';

        if (line === '---') {
            div.className = 'trace-line separator';
        } else if (line.startsWith('⚡') || line.startsWith('🔢') || line.startsWith('📐') || line.startsWith('🎯')) {
            div.className = 'trace-line header-line';
        }

        div.textContent = line === '---' ? '' : line;
        traceList.appendChild(div);
    });

    traceList.scrollTop = 0;

    // Update paradigm badge
    const primary = reasoning.consensus?.primary_paradigm || '—';
    document.getElementById('badge-paradigm').textContent = primary;
}

// ═══════════════════════════════════════
// METACOGNITION GAUGES
// ═══════════════════════════════════════
function setGauge(id, value, labelId) {
    const circumference = 163.36; // 2πr where r=26
    const offset = circumference * (1 - value);
    const circle = document.getElementById(id);
    if (circle) {
        circle.style.strokeDashoffset = offset;
    }
    const label = document.getElementById(labelId);
    if (label) {
        label.textContent = `${(value * 100).toFixed(0)}%`;
    }
}

function updateMetacognition(meta) {
    if (!meta) return;

    const confidence = meta.confidence?.overall || 0;
    const coherence = meta.coherence?.score || 0;
    const emotions = meta.emotional_state || {};
    const curiosity = emotions.curiosity || 0;
    const surprise = meta.surprise?.current_level || 0;

    setGauge('gauge-confidence', confidence, 'gval-confidence');
    setGauge('gauge-coherence', coherence, 'gval-coherence');
    setGauge('gauge-curiosity', curiosity, 'gval-curiosity');
    setGauge('gauge-surprise', surprise, 'gval-surprise');

    // State badge
    const stateText = meta.attention?.focus || 'normal';
    document.getElementById('badge-state').textContent = stateText;
}

// ═══════════════════════════════════════
// GOALS
// ═══════════════════════════════════════
function updateGoals(goalsData) {
    if (!goalsData) return;

    const goalList = document.getElementById('goal-list');
    goalList.innerHTML = '';

    const goals = goalsData.active_goals || [];
    const display = goals.slice(0, 6); // Show max 6

    if (display.length === 0) {
        goalList.innerHTML = '<div class="goal-chip"><span class="goal-priority low"></span><span class="goal-text">No active goals</span></div>';
        return;
    }

    display.forEach(goal => {
        const chip = document.createElement('div');
        chip.className = 'goal-chip';

        const priorityClass = goal.priority > 0.7 ? 'high' : goal.priority > 0.4 ? 'medium' : 'low';

        const typeColors = {
            epistemic: 'badge-cyan',
            homeostatic: 'badge-amber',
            compressive: 'badge-purple',
            instrumental: 'badge-green',
        };

        chip.innerHTML = `
            <span class="goal-priority ${priorityClass}"></span>
            <span class="goal-text">${goal.description}</span>
            <span class="goal-type-badge ${typeColors[goal.type] || 'badge-cyan'}">${goal.type}</span>
        `;
        goalList.appendChild(chip);
    });
}

// ═══════════════════════════════════════
// HEADER STATS
// ═══════════════════════════════════════
function updateHeaderStats(thought) {
    document.getElementById('stat-cycles').textContent = thought.cycle || state.cycles;

    const perception = thought.layers?.perception || {};
    document.getElementById('stat-concepts').textContent = perception.total_concepts_known || 0;

    const memStats = thought.layers?.memory?.stats?.episodic || {};
    document.getElementById('stat-memory').textContent = memStats.total_stored || 0;

    const confidence = thought.layers?.metacognition?.confidence?.overall || 0;
    document.getElementById('stat-confidence').textContent = `${(confidence * 100).toFixed(0)}%`;

    document.getElementById('badge-cycle').textContent = `Cycle ${thought.cycle || state.cycles}`;
}

// ═══════════════════════════════════════
// INIT
// ═══════════════════════════════════════
window.addEventListener('load', () => {
    initGraph();
    chatInput.focus();
});

window.addEventListener('resize', () => {
    initGraph();
    if (state.lastThought) {
        updateGraph(state.lastThought.layers?.world_model);
    }
});
