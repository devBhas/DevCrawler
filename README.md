# 🕷️ DevCrawler- LLM Friendly Web Crawler & Data Scraper

A high-performance web crawler designed for AI/ML training data collection. DevCrawler transforms web content into clean, structured formats (HTML and Markdown) perfect for training Language Models and AI applications. Built with memory efficiency and concurrent processing in mind, it handles JavaScript-heavy sites using Playwright while implementing smart anti-bot detection measures.

✨ First Public Release!

🎉 DevCrawler is now open source! Key features include:
- Concurrent web crawling with configurable workers
- Clean HTML and Markdown output optimized for AI training
- JavaScript rendering support via Playwright
- Memory-efficient processing with smart caching
- Built-in anti-bot detection evasion
- YAML-based configuration and CLI interface

## 🤖 Perfect for AI/ML Projects

- **Training Data Collection**: Generate clean, structured text for model training
- **Content Processing**: Extract and normalize web content automatically
- **Multi-Language Support**: Handle content in various languages
- **Efficient Processing**: Memory-optimized for large-scale crawling
- **Flexible Output**: Both HTML and Markdown formats for different needs

## ✨ Key Features

### 🎯 Content Processing
- Clean HTML and Markdown output
- Semantic content preservation
- Multi-language support
- Configurable crawl depth and breadth

### 🛡️ Browser Automation & Anti-Bot Measures
- JavaScript rendering via Playwright
- Dynamic content extraction
- SPA (Single Page Application) support
- AJAX request handling
- Smart user agent rotation
- Multiple viewport simulation
- Configurable request delays
- Session persistence
- Automatic cookie handling

### ⚡ Performance & Protection
- Memory-efficient processing
- Concurrent crawling with configurable workers
- Smart rate limiting to avoid detection
- Automatic session management
- Intelligent request pacing
- Cache optimization
- Request pattern randomization

### ��️ Configuration
- YAML-based configuration
- Command line interface
- Customizable output formats

## 🚀 Quick Start

1. Clone and install:
```bash
git clone https://github.com/devBhas/DevCrawler.git
cd DevCrawler
pip install -r requirements.txt
python -m playwright install chromium
```

2. Create input.csv: create an input.csv file with the list of websites to crawl.
```csv
website
https://example.com
https://anothersite.com
```

3. Run crawler:
```bash
python crawler.py
```

## 🔧 Configuration

Configure crawler behavior in config.yaml:
```yaml
crawler:
  max_depth: 5                # Maximum crawl depth
  max_urls_per_domain: 5000   # URLs per domain
  rate_limit: 20              # Requests per second
  concurrent_workers: 10      # Parallel workers
  request_timeout: 30         # Timeout in seconds
  respect_robots_txt: false   # robots.txt compliance

browser:
  user_agents:                # Rotating user agents for anti-bot
    - "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124"
    - "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Safari/605.1.15"
  
  viewports:                  # Multiple viewports for detection avoidance
    - {width: 1920, height: 1080}  # Desktop
    - {width: 360, height: 740}    # Mobile

  # Anti-bot measures
  min_delay: 1               # Minimum delay between requests
  max_delay: 5               # Maximum delay between requests
  random_delay: true         # Randomize delays
```

## 📊 Usage Examples

### Basic Crawling
```bash
# Default crawl
python crawler.py

# Set crawl depth
python crawler.py --depth 3

# Limit URLs per domain
python crawler.py --max-urls 1000

# Control crawl speed
python crawler.py --rate-limit 5
```

### Memory Optimization
```bash
# Adjust concurrent workers
python crawler.py --workers 5
```

## 📂 Output Structure

```
DevCrawler-logs/
├── example.com/
│   ├── index.html           # Original HTML
│   ├── index.md            # Clean Markdown
│   └── external_links.csv  # Link analysis
└── scraper_[timestamp].log
```

## 🌐 JavaScript Support

### Current Features
- JavaScript rendering via Playwright
- Dynamic content loading
- AJAX request handling
- Basic SPA support

### Upcoming Features
We're planning several enhancements for future releases:
- Infinite scrolling support
- Click interaction automation
- Form submission handling
- WebSocket data capture
- Custom JavaScript injection
- Advanced browser fingerprinting
- Proxy support
- Custom header configuration

## 👥 Contributing

Contributions welcome! Please feel free to submit issues and pull requests.

## 📄 License

MIT License - see [LICENSE](LICENSE)

## 👤 Author

**Bhaskar Dev (devBhas)**
- GitHub: [@devBhas](https://github.com/devBhas)
- LinkedIn: [@bhaskar-dev](https://www.linkedin.com/in/bhaskar-dev/)
