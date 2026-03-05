f = open('frontend/index.html', 'r', encoding='utf-8')
c = f.read()
f.close()

old = "    const aiOutput = data.output || data.text || JSON.stringify(data);"

new = """    const aiOutput = data.output || data.text || JSON.stringify(data);
    const outputBox = document.getElementById('submitOutput');
    const submitBtn = document.getElementById('submitBtn');
    const outputLabel = document.getElementById('outputLabel');
    outputBox.value = aiOutput;
    outputBox.style.display = 'block';
    if(outputLabel) outputLabel.style.display = 'block';
    if(submitBtn) submitBtn.style.display = 'block';"""

c = c.replace(old, new)

f = open('frontend/index.html', 'w', encoding='utf-8')
f.write(c)
f.close()
print('Done')