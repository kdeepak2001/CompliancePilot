f = open('frontend/index.html', 'r', encoding='utf-8')
c = f.read()
f.close()

home_link = '''<a href="/" style="position:fixed;top:20px;left:20px;z-index:1000;padding:8px 16px;background:rgba(13,21,38,0.9);border:1px solid #1a2540;border-radius:6px;color:#00d4ff;text-decoration:none;font-size:0.8rem;font-family:Courier New,monospace;backdrop-filter:blur(10px);">← Home</a>'''

c = c.replace('<body>', '<body>' + home_link)

f = open('frontend/index.html', 'w', encoding='utf-8')
f.write(c)
f.close()
print('Done')