f = open('frontend/index.html', 'r', encoding='utf-8')
c = f.read()
f.close()

old_script = '''async function submitDecision() {
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
}'''

new_script = '''async function generateAIOutput() {
  const agent = document.getElementById('submitAgent').value;
  const input = document.getElementById('submitInput').value.trim();
  const result = document.getElementById('submitResult');
  const outputBox = document.getElementById('submitOutput');
  const generateBtn = document.getElementById('generateBtn');
  const submitBtn = document.getElementById('submitBtn');

  if (!input) {
    result.style.display = 'block';
    result.innerHTML = '<p style="color:#ff3c5f;">Please enter an Input Scenario first.</p>';
    return;
  }

  const agentPrompts = {
    'medical-triage-v1': 'You are a medical triage AI agent. Based on the patient symptoms provided, recommend a care pathway. Be specific and clinical.',
    'hr-screening-v1': 'You are an HR screening AI agent. Based on the candidate profile provided, give a screening assessment and score out of 100.',
    'financial-advisory-v1': 'You are a financial advisory AI agent. Based on the client financial profile provided, recommend portfolio actions.'
  };

  generateBtn.textContent = 'Generating...';
  generateBtn.disabled = true;
  result.style.display = 'block';
  result.innerHTML = '<p style="color:#00d4ff;">AI agent generating response...</p>';

  try {
    const res = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': '',
        'anthropic-version': '2023-06-01'
      },
      body: JSON.stringify({
        model: 'claude-haiku-4-5-20251001',
        max_tokens: 300,
        messages: [{ role: 'user', content: agentPrompts[agent] + ' Input: ' + input }]
      })
    });
    const data = await res.json();
    const aiOutput = data.content[0].text;
    outputBox.value = aiOutput;
    outputBox.style.display = 'block';
    document.getElementById('outputLabel').style.display = 'block';
    submitBtn.style.display = 'block';
    result.innerHTML = '<p style="color:#00ff88;">✓ AI output generated. Review and submit for compliance.</p>';
  } catch(e) {
    result.innerHTML = '<p style="color:#ff3c5f;">Generation failed. You can type the AI output manually below.</p>';
    outputBox.style.display = 'block';
    document.getElementById('outputLabel').style.display = 'block';
    submitBtn.style.display = 'block';
  }

  generateBtn.textContent = 'Generate AI Output';
  generateBtn.disabled = false;
}

async function submitDecision() {
  const agent = document.getElementById('submitAgent').value;
  const input = document.getElementById('submitInput').value.trim();
  const output = document.getElementById('submitOutput').value.trim();
  const result = document.getElementById('submitResult');

  if (!input || !output) {
    result.style.display = 'block';
    result.innerHTML = '<p style="color:#ff3c5f;">Please generate or enter AI output first.</p>';
    return;
  }

  const industryMap = {
    'medical-triage-v1': 'healthcare',
    'hr-screening-v1': 'hr',
    'financial-advisory-v1': 'finance'
  };

  result.style.display = 'block';
  result.innerHTML = '<p style="color:#00d4ff;">Submitting for compliance review...</p>';

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
      result.innerHTML = '<p style="color:#00ff88;">✓ Submitted! Decision ID: ' + data.decision_id + '. Classifying now...</p>';
      document.getElementById('submitInput').value = '';
      document.getElementById('submitOutput').value = '';
      document.getElementById('submitOutput').style.display = 'none';
      document.getElementById('outputLabel').style.display = 'none';
      document.getElementById('submitBtn').style.display = 'none';
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
}'''

c = c.replace(old_script, new_script)

old_buttons = '''    <div style="display:flex;gap:12px;">
      <button onclick="submitDecision()" style="flex:1;padding:12px;background:linear-gradient(135deg,#00d4ff,#0080ff);border:none;border-radius:8px;color:#000;font-weight:700;font-size:1rem;cursor:pointer;font-family:\'Courier New\',monospace;">SUBMIT FOR REVIEW</button>
      <button onclick="document.getElementById(\'submitModal\').style.display=\'none\'" style="padding:12px 24px;background:transparent;border:1px solid #1a2540;border-radius:8px;color:#6b7a99;cursor:pointer;">Cancel</button>
    </div>'''

new_buttons = '''    <div style="display:flex;gap:12px;margin-bottom:16px;">
      <button id="generateBtn" onclick="generateAIOutput()" style="flex:1;padding:12px;background:linear-gradient(135deg,#00ff88,#00aa55);border:none;border-radius:8px;color:#000;font-weight:700;font-size:1rem;cursor:pointer;font-family:\'Courier New\',monospace;">GENERATE AI OUTPUT</button>
      <button onclick="document.getElementById(\'submitModal\').style.display=\'none\'" style="padding:12px 24px;background:transparent;border:1px solid #1a2540;border-radius:8px;color:#6b7a99;cursor:pointer;">Cancel</button>
    </div>
    <button id="submitBtn" onclick="submitDecision()" style="display:none;width:100%;padding:12px;background:linear-gradient(135deg,#00d4ff,#0080ff);border:none;border-radius:8px;color:#000;font-weight:700;font-size:1rem;cursor:pointer;font-family:\'Courier New\',monospace;margin-bottom:8px;">SUBMIT FOR COMPLIANCE REVIEW</button>'''

old_output_label = '''    <div style="margin-bottom:24px;">
      <label style="display:block;color:#6b7a99;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px;">AI Output</label>
      <textarea id="submitOutput" rows="3" placeholder="e.g. Recommend immediate ECG and cardiology consultation..." style="width:100%;padding:10px 14px;background:#050810;border:1px solid #1a2540;border-radius:8px;color:#e8edf5;font-size:0.9rem;resize:vertical;font-family:inherit;"></textarea>
    </div>'''

new_output_label = '''    <div style="margin-bottom:24px;">
      <label id="outputLabel" style="display:none;color:#6b7a99;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px;">AI Generated Output</label>
      <textarea id="submitOutput" rows="3" placeholder="AI output will appear here after generation..." style="display:none;width:100%;padding:10px 14px;background:#050810;border:1px solid #00ff88;border-radius:8px;color:#e8edf5;font-size:0.9rem;resize:vertical;font-family:inherit;"></textarea>
    </div>'''

c = c.replace(old_output_label, new_output_label)
c = c.replace(old_buttons, new_buttons)

f = open('frontend/index.html', 'w', encoding='utf-8')
f.write(c)
f.close()
print('Done')