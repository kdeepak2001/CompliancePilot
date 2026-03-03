f = open('frontend/landing.html', 'r', encoding='utf-8')
c = f.read()
f.close()

light_css = """
.light { --bg: #f8faff; --bg2: #eef2ff; --surface: #ffffff; --border: #d1d9f0; --text: #0a0f1e; --muted: #5a6480; }
.light body::before { background-image: linear-gradient(rgba(0,100,200,0.04) 1px, transparent 1px), linear-gradient(90deg, rgba(0,100,200,0.04) 1px, transparent 1px); }
.light .hero h1 span { background: linear-gradient(135deg, #0080cc, #0040ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
"""

toggle_btn = '<button onclick="toggleTheme()" style="background:transparent;border:1px solid var(--border);color:var(--muted);padding:8px 16px;border-radius:6px;cursor:pointer;font-size:0.9rem;margin-right:12px;" id="themeBtn">☀️ Light</button>'

toggle_script = """
function toggleTheme() {
  const root = document.documentElement;
  const btn = document.getElementById('themeBtn');
  if (root.classList.contains('light')) {
    root.classList.remove('light');
    btn.textContent = '☀️ Light';
  } else {
    root.classList.add('light');
    btn.textContent = '🌙 Dark';
  }
}
"""

c = c.replace('</style>', light_css + '</style>', 1)
c = c.replace('<a href="/static/index.html" class="nav-cta">', toggle_btn + '<a href="/static/index.html" class="nav-cta">', 1)
c = c.replace('</script>', toggle_script + '</script>', 1)

f = open('frontend/landing.html', 'w', encoding='utf-8')
f.write(c)
f.close()
print('Done')