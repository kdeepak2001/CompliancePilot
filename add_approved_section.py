f = open('frontend/index.html', 'r', encoding='utf-8')
c = f.read()
f.close()

approved_section = """
<div class="card" style="margin-top:24px;">
  <div class="card-header">
    <span class="card-title">APPROVED DECISIONS</span>
    <span id="approvedCount" style="background:rgba(0,255,136,0.1);color:#00ff88;padding:4px 12px;border-radius:20px;font-size:0.75rem;font-family:'Courier New',monospace;">0 APPROVED</span>
  </div>
  <div id="approvedQueue" style="max-height:400px;overflow-y:auto;scroll-behavior:smooth;"></div>
</div>
"""

old_load = "async function loadReviewQueue() {"
new_load = """async function loadApprovedQueue() {
  try {
    const res = await fetch(`${API}/api/decisions?review_status=approved&limit=100`);
    const data = await res.json();
    const approved = data.data || [];
    const count = document.getElementById('approvedCount');
    const queue = document.getElementById('approvedQueue');
    if(count) count.textContent = approved.length + ' APPROVED';
    if(!queue) return;
    if(approved.length === 0) {
      queue.innerHTML = '<div style="padding:20px;text-align:center;color:#6b7a99;font-size:0.85rem;">No approved decisions yet</div>';
      return;
    }
    queue.innerHTML = approved.map(d => `
      <div style="padding:14px 20px;border-bottom:1px solid #1a2540;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
          <span style="color:#00ff88;font-family:'Courier New',monospace;font-size:0.85rem;">#${d.id}</span>
          <span style="background:rgba(0,255,136,0.1);color:#00ff88;padding:2px 10px;border-radius:4px;font-size:0.75rem;">APPROVED</span>
        </div>
        <div style="color:#a0aec0;font-size:0.8rem;margin-bottom:4px;">${d.agent_id}</div>
        <div style="color:#e8edf5;font-size:0.85rem;margin-bottom:6px;">${(d.input_prompt||'').substring(0,80)}...</div>
        <div style="display:flex;gap:16px;font-size:0.75rem;color:#6b7a99;">
          <span>Reviewer: <span style="color:#00d4ff;">${d.reviewer_id || 'N/A'}</span></span>
          <span>Reviewed: <span style="color:#00d4ff;">${d.reviewed_at ? d.reviewed_at.substring(0,10) : 'N/A'}</span></span>
        </div>
        <div style="margin-top:6px;font-size:0.75rem;color:#6b7a99;">Justification: <span style="color:#a0aec0;">${(d.reviewer_justification||'N/A').substring(0,100)}</span></div>
      </div>
    `).join('');
  } catch(e) {
    console.error('Failed to load approved queue', e);
  }
}

async function loadReviewQueue() {"""

c = c.replace(old_load, new_load)

# Add approved section after review queue card
c = c.replace(
    '<!-- SUBMIT MODAL -->',
    approved_section + '<!-- SUBMIT MODAL -->'
)

# Call loadApprovedQueue in loadDashboard
c = c.replace(
    'loadReviewQueue();',
    'loadReviewQueue();\n        loadApprovedQueue();'
)

f = open('frontend/index.html', 'w', encoding='utf-8')
f.write(c)
f.close()
print('Done')

