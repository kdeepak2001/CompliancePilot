f = open('frontend/landing.html', 'r', encoding='utf-8')
c = f.read()
f.close()

# Fix section title size - reduce and add proper spacing
c = c.replace(
    '.section-title { font-family: \'Syne\', sans-serif; font-size: clamp(2rem, 4vw, 3rem); font-weight: 800; letter-spacing: -0.02em; line-height: 1.15; margin-bottom: 20px; }',
    '.section-title { font-family: \'Syne\', sans-serif; font-size: clamp(1.6rem, 3vw, 2.4rem); font-weight: 700; letter-spacing: -0.01em; line-height: 1.25; margin-bottom: 20px; }'
)

# Fix hero title size
c = c.replace(
    'font-size: clamp(3rem, 8vw, 6rem);',
    'font-size: clamp(2.5rem, 5vw, 4.5rem);'
)

# Fix CTA title
c = c.replace(
    'font-size: clamp(2.5rem, 5vw, 4rem);',
    'font-size: clamp(2rem, 3.5vw, 3rem);'
)

# Fix body text line height and size
c = c.replace(
    '.section-desc { color: var(--muted); font-size: 1.1rem; line-height: 1.7; max-width: 600px; margin-bottom: 60px; }',
    '.section-desc { color: var(--muted); font-size: 1rem; line-height: 1.8; max-width: 560px; margin-bottom: 60px; }'
)

# Fix hero subtitle
c = c.replace(
    '.hero-sub { font-size: 1.25rem; color: var(--muted); max-width: 600px; line-height: 1.7; margin-bottom: 48px; font-weight: 300; animation: fadeUp 0.6s ease 0.2s both; }',
    '.hero-sub { font-size: 1.1rem; color: var(--muted); max-width: 560px; line-height: 1.8; margin-bottom: 48px; font-weight: 400; animation: fadeUp 0.6s ease 0.2s both; }'
)

f = open('frontend/landing.html', 'w', encoding='utf-8')
f.write(c)
f.close()
print('Done')

