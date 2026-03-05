f = open('frontend/index.html', 'r', encoding='utf-8')
c = f.read()
f.close()

# Fix timestamp display to IST
c = c.replace(
    "const now = new Date();",
    "const now = new Date(); const istOffset = 5.5 * 60 * 60 * 1000; const istTime = new Date(now.getTime() + istOffset);"
)

c = c.replace(
    "el.textContent = now.toUTCString().replace('GMT', 'UTC');",
    "el.textContent = istTime.toUTCString().replace('GMT', 'IST');"
)

# Fix decision timestamps in feed
c = c.replace(
    "const d = new Date(ts);",
    "const d = new Date(new Date(ts).getTime() + 5.5 * 60 * 60 * 1000);"
)

c = c.replace(
    "return d.toISOString().slice(0,16).replace('T',' ') + ' UTC';",
    "return d.toISOString().slice(0,16).replace('T',' ') + ' IST';"
)

f = open('frontend/index.html', 'w', encoding='utf-8')
f.write(c)
f.close()
print('Done')
