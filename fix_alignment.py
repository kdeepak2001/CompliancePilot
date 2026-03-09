f = open('frontend/landing.html', 'r', encoding='utf-8')
c = f.read()
f.close()

c = c.replace(
    '.problem { padding: 100px 60px; max-width: 1200px; margin: 0 auto; text-align: center; }',
    '.problem { padding: 100px 60px; max-width: 1200px; margin: 0 auto; text-align: left; }'
)

c = c.replace(
    '.section-desc { color: var(--muted); font-size: 1rem; line-height: 1.8; max-width: 560px; margin-bottom: 60px; }',
    '.section-desc { color: var(--muted); font-size: 1rem; line-height: 1.8; max-width: 700px; margin-bottom: 60px; }'
)

f = open('frontend/landing.html', 'w', encoding='utf-8')
f.write(c)
f.close()
print('Done')
