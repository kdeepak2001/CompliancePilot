f = open('frontend/index.html', 'r', encoding='utf-8')
c = f.read()
f.close()

old = "const res = await fetch(`${API}/api/decisions?review_status=pending&limit=10`);"
new = "const res = await fetch(`${API}/api/decisions?review_status=pending&limit=100`);"

c = c.replace(old, new)

f = open('frontend/index.html', 'w', encoding='utf-8')
f.write(c)
f.close()
print('Done')

