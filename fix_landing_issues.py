f = open('frontend/landing.html', 'r', encoding='utf-8')
c = f.read()
f.close()

# Fix 1 - Text alignment under problem heading
c = c.replace(
    '<p class="section-desc">Every AI agent your organization runs is making autonomous decisions right now — with zero documentation, zero oversight records, and zero regulatory evidence.</p>',
    '<p class="section-desc" style="text-align:left;max-width:800px;">Every AI agent your organization runs is making autonomous decisions right now — with zero documentation, zero oversight records, and zero regulatory evidence.</p>'
)

# Fix 2 - DPDP card flag showing IN instead of emoji
c = c.replace(
    '<div class="reg-flag">🇮🇳</div><div class="reg-name">DPDP ACT 2023</div>',
    '<div class="reg-flag" style="font-size:2rem;">🇮🇳</div><div class="reg-name">DPDP ACT 2023</div>'
)

# Fix 3 - Section desc general alignment fix
c = c.replace(
    '.section-desc { color: var(--muted); font-size: 1rem; line-height: 1.8; max-width: 560px; margin-bottom: 60px; }',
    '.section-desc { color: var(--muted); font-size: 1rem; line-height: 1.8; max-width: 800px; margin-bottom: 60px; text-align: left; }'
)

f = open('frontend/landing.html', 'w', encoding='utf-8')
f.write(c)
f.close()
print('Done')
