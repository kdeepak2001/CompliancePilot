f = open('frontend/index.html', 'r', encoding='utf-8')
c = f.read()
f.close()

# Find the decisions table container and add max height with scroll
c = c.replace(
    '<table id="decisionsTable">',
    '<div style="max-height:500px;overflow-y:auto;scroll-behavior:smooth;border-radius:8px;"><table id="decisionsTable">'
)

c = c.replace(
    '</table>\n                </div>\n            </div>',
    '</table></div>\n                </div>\n            </div>'
)

f = open('frontend/index.html', 'w', encoding='utf-8')
f.write(c)
f.close()
print('Done')
