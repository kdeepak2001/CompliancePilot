f = open('frontend/index.html', 'r', encoding='utf-8')
c = f.read()
f.close()

c = c.replace(
    'const aiOutput = data.output;',
    'const aiOutput = data.output || data.text || JSON.stringify(data);'
)

f = open('frontend/index.html', 'w', encoding='utf-8')
f.write(c)
f.close()
print('Done')
