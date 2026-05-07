import React, { useState, useEffect, useRef } from 'react';
import { Activity, BrainCircuit, Terminal, Cpu, MessageSquare } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { motion, AnimatePresence } from 'framer-motion';

const NovaDashboard = () => {
  const [messages, setMessages] = useState([
    { role: 'bot', text: 'NovaCrush Engine Online. GPU acceleration enabled. Awaiting sensory input.' }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [systemState, setSystemState] = useState(null);
  
  const messagesEndRef = useRef(null);
  
  // Auto-scroll chat
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Fetch state periodically
  useEffect(() => {
    const fetchState = async () => {
      try {
        const res = await fetch('http://localhost:5000/api/state');
        if (res.ok) {
          const data = await res.json();
          setSystemState(data);
        }
      } catch (e) {
        // Ignore silent errors for now to not spam console if server is down
      }
    };
    
    fetchState();
    const interval = setInterval(fetchState, 3000);
    return () => clearInterval(interval);
  }, []);

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return;
    
    const userMsg = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', text: userMsg }]);
    setIsLoading(true);
    
    try {
      const res = await fetch('http://localhost:5000/api/think', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ input: userMsg })
      });
      
      const data = await res.json();
      
      if (data.error) {
        setMessages(prev => [...prev, { role: 'bot', error: true, text: `API Error: ${data.error}` }]);
      } else {
        const confidence = data.response.confidence ? Math.round(data.response.confidence * 100) : 0;
        const text = data.response.text || "No response generated.";
        
        // Extract native string and topological info if possible
        const lines = text.split('\n');
        const nativeResponse = lines.find(l => l.includes('NovaCrush Native Response:'))?.replace('**NovaCrush Native Response:**', '').trim() || text;
        const topological = lines.find(l => l.includes('Topological Prediction:')) || '';
        
        setMessages(prev => [...prev, { 
          role: 'bot', 
          text: nativeResponse,
          meta: `Cycle #${data.cycle} | Conf: ${confidence}% | ${topological}`
        }]);
      }
    } catch (e) {
      setMessages(prev => [...prev, { role: 'bot', error: true, text: 'Connection to AGI Engine failed.' }]);
    } finally {
      setIsLoading(false);
    }
  };

  // Mock data for curriculum chart until we have real streaming data
  const trainingData = [
    { step: 0, surprise: 0.95 },
    { step: 200, surprise: 0.88 },
    { step: 400, surprise: 0.72 },
    { step: 600, surprise: 0.65 },
    { step: 800, surprise: 0.58 },
    { step: 1000, surprise: 0.51 }
  ];

  return (
    <div className="flex h-screen p-4 gap-4 bg-background text-slate-200 font-sans">
      
      {/* LEFT PANEL: Geometric Substrate & Curriculum */}
      <div className="flex flex-col w-1/4 gap-4">
        {/* Substrate Viz */}
        <div className="flex-1 bg-surface border border-border rounded-xl p-4 flex flex-col relative overflow-hidden backdrop-blur-md">
          <div className="flex items-center gap-2 mb-4 text-neonPurple font-semibold">
            <BrainCircuit size={18} />
            <span>Geometric Substrate</span>
          </div>
          
          <div className="flex-1 relative flex items-center justify-center">
            {/* Fake 3D Node Graph Effect */}
            <div className="absolute inset-0 opacity-50 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-neonPurple/20 via-transparent to-transparent"></div>
            <div className="text-center z-10">
              <div className="text-3xl font-bold text-white mb-1">
                {systemState?.perception?.total_concepts || 4000}
              </div>
              <div className="text-xs text-slate-400 uppercase tracking-widest">Active Concepts</div>
              <div className="mt-4 text-xs font-mono text-neonPurple/70">
                Dim: 768 <br/> HDC: 4096
              </div>
            </div>
            
            {/* Decorative particles */}
            {[...Array(15)].map((_, i) => (
              <motion.div
                key={i}
                className="absolute w-1 h-1 bg-neonPurple rounded-full shadow-[0_0_10px_#8b5cf6]"
                animate={{
                  x: [Math.random() * 200 - 100, Math.random() * 200 - 100],
                  y: [Math.random() * 200 - 100, Math.random() * 200 - 100],
                  opacity: [0.2, 1, 0.2]
                }}
                transition={{ duration: Math.random() * 5 + 5, repeat: Infinity, ease: "linear" }}
              />
            ))}
          </div>
        </div>

        {/* Curriculum Training Chart */}
        <div className="h-64 bg-surface border border-border rounded-xl p-4 flex flex-col backdrop-blur-md">
          <div className="flex items-center gap-2 mb-4 text-accent font-semibold">
            <Activity size={18} />
            <span>Curriculum Training</span>
          </div>
          <div className="flex-1 -ml-4">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={trainingData}>
                <XAxis dataKey="step" stroke="#475569" fontSize={10} tickFormatter={v => `${v}s`} />
                <YAxis stroke="#475569" fontSize={10} domain={[0, 1]} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b' }}
                  itemStyle={{ color: '#3b82f6' }}
                />
                <Line type="monotone" dataKey="surprise" stroke="#3b82f6" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* CENTER PANEL: Chat Interface */}
      <div className="flex-1 bg-surface border border-border rounded-xl flex flex-col backdrop-blur-md shadow-2xl">
        <div className="p-4 border-b border-border flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
            <h1 className="font-bold text-lg tracking-wide">NovaCrush Interface</h1>
          </div>
          <div className="flex items-center gap-4 text-xs text-slate-400 font-mono">
            <div className="flex items-center gap-1"><Cpu size={14}/> RTX 3050</div>
            <div className="flex items-center gap-1">Cycle {systemState?.cycle_count || 0}</div>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-6 flex flex-col gap-6">
          <AnimatePresence>
            {messages.map((msg, idx) => (
              <motion.div 
                key={idx}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className={`max-w-[85%] flex flex-col ${msg.role === 'user' ? 'self-end items-end' : 'self-start items-start'}`}
              >
                <div className={`px-5 py-3 rounded-2xl ${
                  msg.role === 'user' 
                    ? 'bg-accent/20 border border-accent/30 text-white rounded-br-none' 
                    : msg.error 
                      ? 'bg-red-500/10 border border-red-500/30 text-red-200 rounded-bl-none'
                      : 'bg-white/5 border border-white/10 text-slate-100 rounded-bl-none'
                }`}>
                  {msg.text}
                </div>
                {msg.meta && (
                  <div className="text-[10px] text-slate-500 mt-1.5 font-mono px-2">
                    {msg.meta}
                  </div>
                )}
              </motion.div>
            ))}
            {isLoading && (
              <motion.div 
                initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                className="self-start text-xs text-accent animate-pulse font-mono flex items-center gap-2"
              >
                <Activity size={12} /> Synthesizing geometric trajectory...
              </motion.div>
            )}
          </AnimatePresence>
          <div ref={messagesEndRef} />
        </div>

        <div className="p-4 border-t border-border bg-black/20">
          <div className="flex gap-2">
            <input
              type="text"
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && sendMessage()}
              placeholder="Inject sensory text sequence..."
              className="flex-1 bg-white/5 border border-white/10 rounded-lg px-4 py-3 text-sm focus:outline-none focus:border-accent transition-colors text-white placeholder-slate-500"
              disabled={isLoading}
            />
            <button 
              onClick={sendMessage}
              disabled={isLoading || !input.trim()}
              className="bg-accent hover:bg-accent/80 text-white px-6 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
            >
              <MessageSquare size={18} />
            </button>
          </div>
        </div>
      </div>

      {/* RIGHT PANEL: Broca's Area Terminal */}
      <div className="w-1/4 bg-black/40 border border-border rounded-xl p-4 flex flex-col backdrop-blur-md font-mono text-xs">
        <div className="flex items-center gap-2 mb-4 text-emerald-400 font-semibold border-b border-border pb-3">
          <Terminal size={18} />
          <span>Broca's Area Log</span>
        </div>
        
        <div className="flex-1 overflow-y-auto text-slate-400 space-y-2">
          <div className="text-emerald-500/50">{'>>'} Initialization complete</div>
          <div className="text-emerald-500/50">{'>>'} Loaded 4000 core concepts</div>
          <div className="text-emerald-500/50">{'>>'} Transition matrix ready</div>
          {messages.filter(m => m.role === 'bot' && !m.error).map((msg, i) => (
            <div key={i} className="pt-2 border-t border-white/5">
              <span className="text-accent">{'[GEN]'}</span> Walking substrate trajectory...<br/>
              <span className="text-slate-300">"{msg.text}"</span>
            </div>
          ))}
        </div>
      </div>

    </div>
  );
};

export default NovaDashboard;
