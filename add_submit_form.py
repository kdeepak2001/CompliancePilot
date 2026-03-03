f = open('frontend/index.html', 'r', encoding='utf-8')
c = f.read()
f.close()

fab_html = """
<!-- FLOATING SUBMIT BUTTON -->
<button onclick="document.getElementById('submitModal').style.display='flex'" style="position:fixed;bottom:32px;right:32px;z-index:1000;width:56px;height:56px;border-radius:50%;background:linear-gradient(135deg,#00d4ff,#0080ff);border:none;cursor:pointer;font-size:1.5rem;box-shadow:0 8px 32px rgba(0,212,255,0.4);transition:all 0.2s;" title="Submit Decision">+</button>

<!-- SUBMIT MODAL -->
<div id="submitModal" style="display:none;position:fixed;inset:0;z-index:2000;background:rgba(0,0,0,0.8);backdrop-filter:blur(8px);align-items:center;justify-content:center;padding:20px;">
  <div style="background:#0d1526;border:1px solid #1a2540;border-radius:16px;padding:40px;width:100%;max-width:600px;position:relative;">
    <button onclick="document.getElementById('submitModal').style.display='none'" style="position:absolute;top:16px;right:16px;background:transparent;border:none;color:#6b7a99;font-size:1.5rem;cursor:pointer;">×</button>
    <h2 style="font-family:'Courier New',monospace;color:#00d4ff;font-size:1.3rem;margin-bottom:8px;">SUBMIT DECISION</h2>
    <p style="color:#6b7a99;font-size:0.85rem;margin-bottom:24px;">Submit an AI agent decision for compliance classification</p>
    
    <div style="margin-bottom:16px;">
      <label style="display:block;color:#6b7a99;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px;">Agent</label>
      <select id="submitAgent" style="width:100%;padding:10px 14px;background:#050810;border:1px solid #1a2540;border-radius:8px;color:#e8edf5;font-size:0.9rem;">
        <option value="medical-triage-v1">Medical Triage Agent</option>
        <option value="hr-screening-v1">HR Screening Agent</option>
        <option value="financial-advisory-v1">Financial Advisory Agent</option>
      </select>
    </div>

    <div style="margin-bottom:16px;">
      <label style="display:block;color:#6b7a99;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px;">Input Scenario</label>
      <textarea id="submitInput" rows="3" placeholder="e.g. Patient is a 65 year old female with severe chest pain..." style="width:100%;padding:10px 14px;background:#050810;border:1px solid #1a2540;border-radius:8px;color:#e8edf5;font-size:0.9rem;resize:vertical;font-family:inherit;"></textarea>
    </div>

    <div style="margin-bottom:24px;">
      <label style="display:block;color:#6b7a99;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px;">AI Output</label>
      <textarea id="submitOutput" rows="3" placeholder="e.g. Recommend immediate ECG and cardiology consultation..." style="width:100%;padding:10px 14px;background:#050810;border:1px solid #1a2540;border-radius:8px;color:#e8edf5;font-size:0.9rem;resize:vertical;font-family:inherit;"></textarea>
    </div>

    <div style="display:flex;gap:12px;">
      <button onclick="submitDecision()" style="flex:1;padding:12px;background:linear-gradient(135deg,#00d4ff,#0080ff);border:none;border-radius:8px;color:#000;font-weight:700;font-size:1rem;cursor:pointer;font-family:'Courier New',monospace;">SUBMIT FOR REVIEW</button>
      <button onclick="document.getElementById('submitModal').style.display='none'" style="padding:12px 24px;background:transparent;border:1px solid #1a2540;border-radius:8px;color:#6b7a99;cursor:pointer;">Cancel</button>
    </div>
    <div id="submitResult" style="margin-top:16px;display:none;"></div>
  </div>
</div>

<script>
async function submitDecision() {
  const agent = document.getElementById('submitAgent').value;
  const input = document.getElementById('submitInput').value.trim();
  const output = document.getElementById('submitOutput').value.trim();
  const result = document.getElementById('submitResult');

  if (!input || !output) {
    result.style.display = 'block';
    result.innerHTML = '<p style="color:#ff3c5f;">Please fill in both Input Scenario and AI Output.</p>';
    return;
  }

  const industryMap = {
    'medical-triage-v1': 'healthcare',
    'hr-screening-v1': 'hr',
    'financial-advisory-v1': 'finance'
  };

  result.style.display = 'block';
  result.innerHTML = '<p style="color:#00d4ff;">Submitting and classifying...</p>';

  try {
    const res = await fetch(`${API}/api/decisions/log`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: 'session-' + Date.now(),
        agent_id: agent,
        agent_version: '1.0.0',
        input_prompt: input,
        agent_output: output,
        decision_context: JSON.stringify({ industry: industryMap[agent] }),
        processing_time_ms: 1000,
        is_demo: true
      })
    });
    const data = await res.json();
    if (data.success) {
      result.innerHTML = '<p style="color:#00ff88;">✓ Decision submitted! ID: ' + data.decision_id + '. Classification in progress...</p>';
      document.getElementById('submitInput').value = '';
      document.getElementById('submitOutput').value = '';
      setTimeout(() => {
        document.getElementById('submitModal').style.display = 'none';
        result.style.display = 'none';
        loadDashboard();
      }, 3000);
    } else {
      result.innerHTML = '<p style="color:#ff3c5f;">Error: ' + JSON.stringify(data) + '</p>';
    }
  } catch(e) {
    result.innerHTML = '<p style="color:#ff3c5f;">Connection error. Try again.</p>';
  }
}
</script>
"""

c = c.replace('</body>', fab_html + '</body>')

f = open('frontend/index.html', 'w', encoding='utf-8')
f.write(c)
f.close()
print('Done')