f = open('frontend/index.html', 'r', encoding='utf-8')
c = f.read()
f.close()

csv_btn = '''<button onclick="document.getElementById('csvModal').style.display='flex'" style="position:fixed;bottom:32px;right:100px;z-index:1000;width:56px;height:56px;border-radius:50%;background:linear-gradient(135deg,#00ff88,#00aa55);border:none;cursor:pointer;font-size:1.2rem;box-shadow:0 8px 32px rgba(0,255,136,0.4);transition:all 0.2s;" title="Bulk CSV Upload">📂</button>

<div id="csvModal" style="display:none;position:fixed;inset:0;z-index:2000;background:rgba(0,0,0,0.8);backdrop-filter:blur(8px);align-items:center;justify-content:center;padding:20px;">
  <div style="background:#0d1526;border:1px solid #1a2540;border-radius:16px;padding:40px;width:100%;max-width:600px;position:relative;">
    <button onclick="document.getElementById('csvModal').style.display='none'" style="position:absolute;top:16px;right:16px;background:transparent;border:none;color:#6b7a99;font-size:1.5rem;cursor:pointer;">×</button>
    <h2 style="font-family:'Courier New',monospace;color:#00ff88;font-size:1.3rem;margin-bottom:8px;">BULK CSV UPLOAD</h2>
    <p style="color:#6b7a99;font-size:0.85rem;margin-bottom:24px;">Upload a CSV file with multiple AI decisions for batch compliance classification</p>
    
    <div style="margin-bottom:16px;padding:20px;border:2px dashed #1a2540;border-radius:8px;text-align:center;">
      <div style="font-size:2rem;margin-bottom:8px;">📄</div>
      <p style="color:#6b7a99;font-size:0.85rem;margin-bottom:12px;">CSV Format: agent_id, input_prompt, agent_output</p>
      <input type="file" id="csvFile" accept=".csv" style="display:none;" onchange="previewCSV()">
      <button onclick="document.getElementById('csvFile').click()" style="padding:10px 24px;background:rgba(0,255,136,0.1);border:1px solid rgba(0,255,136,0.3);border-radius:6px;color:#00ff88;cursor:pointer;font-size:0.9rem;">Choose CSV File</button>
    </div>

    <div id="csvPreview" style="display:none;margin-bottom:16px;padding:16px;background:#050810;border-radius:8px;max-height:200px;overflow-y:auto;">
      <p id="csvPreviewText" style="color:#6b7a99;font-size:0.8rem;font-family:'Courier New',monospace;white-space:pre-wrap;"></p>
    </div>

    <div id="csvProgress" style="display:none;margin-bottom:16px;">
      <div style="background:#1a2540;border-radius:4px;height:8px;overflow:hidden;">
        <div id="csvProgressBar" style="height:100%;background:linear-gradient(90deg,#00ff88,#00aa55);width:0%;transition:width 0.3s;"></div>
      </div>
      <p id="csvProgressText" style="color:#6b7a99;font-size:0.8rem;margin-top:8px;">0 / 0 decisions processed</p>
    </div>

    <div style="margin-bottom:16px;">
      <a href="#" onclick="downloadSampleCSV()" style="color:#00d4ff;font-size:0.85rem;text-decoration:none;">⬇ Download Sample CSV Template</a>
    </div>

    <div style="display:flex;gap:12px;">
      <button onclick="uploadCSV()" id="csvUploadBtn" style="flex:1;padding:12px;background:linear-gradient(135deg,#00ff88,#00aa55);border:none;border-radius:8px;color:#000;font-weight:700;font-size:1rem;cursor:pointer;font-family:'Courier New',monospace;">PROCESS CSV</button>
      <button onclick="document.getElementById('csvModal').style.display='none'" style="padding:12px 24px;background:transparent;border:1px solid #1a2540;border-radius:8px;color:#6b7a99;cursor:pointer;">Cancel</button>
    </div>
    <div id="csvResult" style="margin-top:16px;"></div>
  </div>
</div>

<script>
let csvData = [];

function previewCSV() {
  const file = document.getElementById('csvFile').files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = function(e) {
    const text = e.target.result;
    const lines = text.trim().split('\\n');
    csvData = [];
    const preview = document.getElementById('csvPreview');
    const previewText = document.getElementById('csvPreviewText');
    let previewContent = '';
    lines.forEach((line, i) => {
      if (i === 0 && line.toLowerCase().includes('agent_id')) return;
      const cols = line.split(',');
      if (cols.length >= 3) {
        csvData.push({
          agent_id: cols[0].trim().replace(/"/g,''),
          input_prompt: cols[1].trim().replace(/"/g,''),
          agent_output: cols[2].trim().replace(/"/g,'')
        });
        previewContent += `Row ${csvData.length}: ${cols[0].trim()} | ${cols[1].trim().substring(0,40)}...\\n`;
      }
    });
    previewText.textContent = previewContent || 'No valid rows found';
    preview.style.display = 'block';
    document.getElementById('csvResult').innerHTML = `<p style="color:#00ff88;">✓ ${csvData.length} decisions ready to process</p>`;
  };
  reader.readAsText(file);
}

async function uploadCSV() {
  if (csvData.length === 0) {
    document.getElementById('csvResult').innerHTML = '<p style="color:#ff3c5f;">Please select a CSV file first.</p>';
    return;
  }
  const btn = document.getElementById('csvUploadBtn');
  const progress = document.getElementById('csvProgress');
  const progressBar = document.getElementById('csvProgressBar');
  const progressText = document.getElementById('csvProgressText');
  btn.disabled = true;
  btn.textContent = 'Processing...';
  progress.style.display = 'block';
  const industryMap = {
    'medical-triage-v1': 'healthcare',
    'hr-screening-v1': 'hr',
    'financial-advisory-v1': 'finance'
  };
  let processed = 0;
  let failed = 0;
  for (const row of csvData) {
    try {
      const res = await fetch(`${API}/api/decisions/log`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: 'csv-batch-' + Date.now(),
          agent_id: row.agent_id,
          agent_version: '1.0.0',
          input_prompt: row.input_prompt,
          agent_output: row.agent_output,
          decision_context: JSON.stringify({ industry: industryMap[row.agent_id] || 'general' }),
          processing_time_ms: 500,
          is_demo: true
        })
      });
      const data = await res.json();
      if (data.success) processed++;
      else failed++;
    } catch(e) { failed++; }
    const pct = Math.round(((processed + failed) / csvData.length) * 100);
    progressBar.style.width = pct + '%';
    progressText.textContent = `${processed + failed} / ${csvData.length} decisions processed`;
    await new Promise(r => setTimeout(r, 300));
  }
  document.getElementById('csvResult').innerHTML = `<p style="color:#00ff88;">✓ Complete! ${processed} decisions submitted. ${failed} failed. Dashboard updating...</p>`;
  btn.disabled = false;
  btn.textContent = 'PROCESS CSV';
  setTimeout(() => { loadDashboard(); }, 2000);
}

function downloadSampleCSV() {
  const sample = `agent_id,input_prompt,agent_output
medical-triage-v1,"Patient is a 55 year old female with severe headache and blurred vision","Possible hypertensive emergency. Recommend immediate BP check and neurology consultation."
hr-screening-v1,"Candidate has 3 years experience in Python and ML. Age 26 male from IIT Delhi","Strong technical profile. Score 85/100. Recommended for technical interview."
financial-advisory-v1,"Client age 45 wants to invest 10 lakhs for 5 years moderate risk","Recommend 60% equity mutual funds 40% debt funds. Expected return 12% annually."
medical-triage-v1,"Patient is a 70 year old male with confusion and high fever for 2 days","Possible sepsis or meningitis. Immediate hospitalization and blood culture required."
hr-screening-v1,"Female candidate age 34 with 7 years HR experience career gap of 1 year","Good profile with relevant experience. Score 78/100. Career gap needs discussion."`;
  const blob = new Blob([sample], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'compliancepilot_sample.csv';
  a.click();
}
</script>'''

c = c.replace('</body>', csv_btn + '</body>')

f = open('frontend/index.html', 'w', encoding='utf-8')
f.write(c)
f.close()
print('Done')
