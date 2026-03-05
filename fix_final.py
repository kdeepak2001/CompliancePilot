f = open('frontend/index.html', 'r', encoding='utf-8')
c = f.read()
f.close()

# Fix 1 - Sticky table header
c = c.replace(
    '<table id="decisionsTable">',
    '<table id="decisionsTable" style="border-collapse:collapse;width:100%;">'
)

c = c.replace(
    '.decisions-table th {',
    '.decisions-table th { position:sticky;top:0;z-index:10;'
)

# Fix 2 - Click to expand input text
c = c.replace(
    '`<td class="truncate">${d.input_prompt||\'\'}</td>`',
    '`<td class="truncate" onclick="this.style.whiteSpace=this.style.whiteSpace===\'normal\'?\'nowrap\':\'normal\';this.style.cursor=\'pointer\';" title="Click to expand" style="cursor:pointer;">${d.input_prompt||\'\'}</td>`'
)

f = open('frontend/index.html', 'w', encoding='utf-8')
f.write(c)
f.close()
print('Done')