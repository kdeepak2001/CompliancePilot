f = open('frontend/index.html', 'r', encoding='utf-8')
c = f.read()
f.close()

old_fetch = """  try {
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
    const aiOutput = data.content[0].text;"""

new_fetch = """  try {
    const res = await fetch(`${API}/api/generate-output`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ agent_id: agent, input_prompt: input })
    });
    const data = await res.json();
    const aiOutput = data.output;"""

c = c.replace(old_fetch, new_fetch)

f = open('frontend/index.html', 'w', encoding='utf-8')
f.write(c)
f.close()
print('Done')