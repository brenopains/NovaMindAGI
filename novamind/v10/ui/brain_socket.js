const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
const ws = new WebSocket(`${protocol}//${location.host}/ws`);

const logDiv = document.getElementById('chat-log');
const inputField = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const sysStatus = document.getElementById('sys-status');
const statusWrapper = document.querySelector('.status-indicator');

// Metrics DOM
const moeIdx = document.getElementById('moe-idx');
const symbProg = document.getElementById('symb-prog');
const symbAst = document.getElementById('symb-ast');
const memIdx = document.getElementById('mem-idx');
const lossIndicator = document.getElementById('loss-indicator');
const efeBars = [
    document.getElementById('b0'),
    document.getElementById('b1'),
    document.getElementById('b2'),
    document.getElementById('b3'),
    document.getElementById('b4')
];
const spikeFlashes = [
    document.getElementById('s0'),
    document.getElementById('s1'),
    document.getElementById('s2'),
    document.getElementById('s3'),
    document.getElementById('s4')
];

// --- WebGL 3D PointCloud (NovaMind <-> Lyra Simulated Environment) ---
let scene, camera, renderer, particles, particleGeo, particleMat;

function init3DWorld() {
    const container = document.getElementById('webgl-container');
    if (!container) return; // Will wait for HTML update
    
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
    renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    // Generate a Dummy Gaussian Splat field
    particleGeo = new THREE.BufferGeometry();
    const particleCount = 2000;
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);

    for (let i = 0; i < particleCount * 3; i+=3) {
        // Spherical distribution around center
        positions[i] = (Math.random() - 0.5) * 10;
        positions[i+1] = (Math.random() - 0.5) * 10;
        positions[i+2] = (Math.random() - 0.5) * 10;
        
        // Colors mapping to Lyra's latent aesthetics
        colors[i] = Math.random() * 0.2; // R
        colors[i+1] = 0.7 + Math.random() * 0.3; // G (cyan focus)
        colors[i+2] = 0.8 + Math.random() * 0.2; // B
    }

    particleGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    particleGeo.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    particleMat = new THREE.PointsMaterial({
        size: 0.1,
        vertexColors: true,
        transparent: true,
        opacity: 0.7
    });

    particles = new THREE.Points(particleGeo, particleMat);
    scene.add(particles);

    camera.position.z = 5;

    // Animation Loop
    function animate() {
        requestAnimationFrame(animate);
        particles.rotation.y += 0.002;
        particles.rotation.x += 0.001;
        renderer.render(scene, camera);
    }
    animate();
}

window.addEventListener('resize', () => {
    if(camera && renderer) {
        const container = document.getElementById('webgl-container');
        camera.aspect = container.clientWidth / container.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(container.clientWidth, container.clientHeight);
    }
});

// Initialize 3D on load
setTimeout(init3DWorld, 500);

function appendMsg(text, type='sys') {
    const d = document.createElement('div');
    d.className = `msg ${type}`;
    d.textContent = text;
    logDiv.appendChild(d);
    logDiv.scrollTop = logDiv.scrollHeight;
}

// --- Multimedia & Upload Logic ---
const fileBtn = document.getElementById('file-btn');
const fileInput = document.getElementById('file-input');
const micBtn = document.getElementById('mic-btn');

let activeMediaB64 = null;
let activeMediaType = null;

if (fileBtn && fileInput) {
    fileBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;
        
        const reader = new FileReader();
        reader.onload = (ev) => {
            // Split base64 header ("data:image/png;base64,") for cleaner backend passing
            // Or just pass the entire string. We will pass the full string and parse via API.
            activeMediaB64 = ev.target.result; 
            activeMediaType = file.type;
            appendMsg(`[ATTACHED] ${file.name}`, 'sys');
        };
        reader.readAsDataURL(file);
    });
}

// Voice Chat Logic
let mediaRecorder;
let audioChunks = [];
let isRecording = false;

if (micBtn) {
    micBtn.addEventListener('click', async () => {
        if (!isRecording) {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                
                mediaRecorder.ondataavailable = e => {
                    if(e.data.size > 0) audioChunks.push(e.data);
                };
                
                mediaRecorder.onstop = () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                    audioChunks = [];
                    const reader = new FileReader();
                    reader.onload = (ev) => {
                        activeMediaB64 = ev.target.result;
                        activeMediaType = 'audio/webm';
                        appendMsg(`[AUDIO CAPTURED] Transmitting...`, 'user');
                        sendInput(); // Auto send after recording
                    };
                    reader.readAsDataURL(audioBlob);
                    stream.getTracks().forEach(track => track.stop());
                };
                
                mediaRecorder.start();
                isRecording = true;
                micBtn.classList.add('recording');
            } catch (err) {
                appendMsg(`[MIC ERROR] ${err}`, 'sys');
            }
        } else {
            mediaRecorder.stop();
            isRecording = false;
            micBtn.classList.remove('recording');
        }
    });
}

function sendInput() {
    const text = inputField.value.trim();
    if (!text && !activeMediaB64) return;
    
    if (text) appendMsg(`> ${text}`, 'user');
    
    const payload = {
        type: "sensory_input",
        text: text,
        media: activeMediaB64,
        media_type: activeMediaType
    };
    
    if (ws.readyState !== WebSocket.OPEN) {
        appendMsg(`[ERROR] Neural Link Server is offline/disconnected.`, 'sys');
        return;
    }

    ws.send(JSON.stringify(payload));
    
    inputField.value = '';
    activeMediaB64 = null;
    activeMediaType = null;
    fileInput.value = '';
}

sendBtn.addEventListener('click', sendInput);
inputField.addEventListener('keypress', e => {
    if(e.key === 'Enter') sendInput();
});

ws.onopen = () => {
    sysStatus.textContent = "SYS.ONLINE";
    appendMsg("Neural link connected on Port 8000.", "sys");
};

ws.onmessage = (event) => {
    const payload = JSON.parse(event.data);
    
    if (payload.type === 'status' && payload.message === 'processing') {
        sysStatus.textContent = "COGNITION.ACTIVE";
        statusWrapper.classList.add('active');
        statusWrapper.querySelector('.dot').classList.add('pulse-anim');
    }
    
    if (payload.type === 'brain_cycle') {
        const data = payload.data;
        
        // Return UI to idle
        sysStatus.textContent = "SYS.IDLE";
        statusWrapper.classList.remove('active');
        statusWrapper.querySelector('.dot').classList.remove('pulse-anim');
        
        // Update Panel Metrics
        moeIdx.textContent = `EXPERT_${data.moe_expert_idx.toString().padStart(2, '0')}`;
        
        // Setup Visual AST
        symbAst.innerHTML = ''; // clear
        symbAst.appendChild(symbProg); // Keep the main label
        symbProg.textContent = data.symbolic_prog || '[NO_OP]';
        
        if(data.symbolic_ast && data.symbolic_ast.nodes) {
            data.symbolic_ast.nodes.forEach((node, idx) => {
                const el = document.createElement('div');
                el.className = 'ast-node';
                el.textContent = node.label;
                symbAst.appendChild(el);
                if (idx < data.symbolic_ast.nodes.length - 1) {
                    const arrow = document.createElement('div');
                    arrow.className = 'ast-edge';
                    arrow.textContent = '→';
                    symbAst.appendChild(arrow);
                }
            });
        }

        // Update 3D Camera navigation based on Action Tensor!
        // This is the NovaMind bridging into Lyra's generative trajectory 
        if(camera && data.action_vector) {
            // Morph camera projection smoothly towards Tensor intention
            const targetZ = 3 + (data.action_vector[1] * 2);
            const targetX = data.action_vector[0] * 3;
            // Tween camera
            camera.position.z += (targetZ - camera.position.z) * 0.1;
            camera.position.x += (targetX - camera.position.x) * 0.1;
            
            // Perturb points scale by loss values if training
            if(particleMat && data.loss_val) {
               particleMat.size = 0.05 + Math.min(0.2, data.loss_val * 0.02);
            }
        }

        memIdx.textContent = `0x${data.mem_index.toString(16).toUpperCase()}`;
        
        // Show Real Training Loss
        if (data.loss_val !== undefined && data.loss_val > 0) {
            lossIndicator.textContent = `Optimizing Loss: ${data.loss_val.toFixed(4)}`;
        }
        
        // Update Neuromorphic LIF Action Meters (Membrane Potentials)
        if(data.membrane_potentials) {
            data.membrane_potentials.forEach((val, i) => {
                if(i < efeBars.length) {
                    // Neural voltage grows as it nears threshold (1.0 typical)
                    const normalized = Math.min(100, Math.max(5, Math.abs(val) * 100));
                    efeBars[i].style.height = `${normalized}%`;
                }
            });
        }
        
        // Flash UI on Spikes! Biological Firing Pattern
        if(data.spikes) {
            let spiked = false;
            let spkSig = "";
            data.spikes.forEach((val, i) => {
                if(i < spikeFlashes.length && val > 0) {
                    // Instant Flash
                    spikeFlashes[i].classList.add('fired');
                    setTimeout(() => spikeFlashes[i].classList.remove('fired'), 50);
                    spiked = true;
                }
                spkSig += val > 0 ? "1" : "0";
            });
            if(spiked) {
                appendMsg(`⚡ [SNN_SPIKE] Pattern Fired: ${spkSig}`, 'sys');
            }
        }
        
        if (!data.is_autonomous && data.emitted_word) {
            appendMsg(`>> ${data.emitted_word}`, 'agi');
        } else if (data.loss_val && !data.spikes?.includes(1)) {
            // Optional: log passive backprop
            // appendMsg(`[Backprop] Target Word vs Emitted: -> [${data.emitted_word}] (Loss: ${data.loss_val.toFixed(4)})`, 'sys');
        }
    }
};

ws.onclose = () => {
    sysStatus.textContent = "SYS.DISCONNECTED";
    appendMsg("Connection lost to Core.", "sys");
};
