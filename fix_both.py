# Fix 1 - Landing page logo spacing
f = open('frontend/landing.html', 'r', encoding='utf-8')
c = f.read()
f.close()
c = c.replace(
    '.nav-logo { display: flex; align-items: center; gap: 12px; font-family: \'Syne\', sans-serif; font-weight: 800; font-size: 1.3rem; }',
    '.nav-logo { display: flex; align-items: center; gap: 12px; font-family: \'Syne\', sans-serif; font-weight: 800; font-size: 1.3rem; letter-spacing: 0.03em; }'
)
f = open('frontend/landing.html', 'w', encoding='utf-8')
f.write(c)
f.close()
print('Landing page fixed')

# Fix 2 - Move back button so it does not cover logo on dashboard
f = open('frontend/index.html', 'r', encoding='utf-8')
c = f.read()
f.close()
c = c.replace(
    'style="position:fixed;top:20px;left:20px;z-index:1000;',
    'style="position:fixed;top:20px;left:50%;transform:translateX(-50%);z-index:1000;'
)
f = open('frontend/index.html', 'w', encoding='utf-8')
f.write(c)
f.close()
print('Dashboard fixed')
