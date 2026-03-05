f = open('frontend/index.html', 'r', encoding='utf-8')
c = f.read()
f.close()

# Find the review queue container and add proper scrolling
c = c.replace(
    'REVIEW QUEUE',
    'REVIEW QUEUE'
)

# Fix the review queue CSS to allow proper scrolling
c = c.replace(
    '.review-queue {',
    '.review-queue { max-height: 600px; overflow-y: auto; scroll-behavior: smooth;'
)

f = open('frontend/index.html', 'w', encoding='utf-8')
f.write(c)
f.close()
print('Done')
