import feedparser

url = "https://www.nrk.no/nyheter/siste.rss"

# Parse the RSS feed
feed = feedparser.parse(url)

# Check if the feed was fetched successfully
if feed.bozo == 0 or len(feed.entries) > 0:
    print(f"Feed Title: {feed.feed.title}\n" + "="*40)
    
    # Iterate through the articles (entries)
    for entry in feed.entries:
        print(f"Title: {entry.title}")
        print(f"Link: {entry.link}")
        # Use .get() for optional fields to avoid KeyErrors
        print(f"Published: {entry.get('published', 'No date provided')}") 
        print(f"Summary: {entry.get('description', 'No summary provided')}")
        print("-" * 40)
else:
    print("Failed to fetch or parse the RSS feed.")