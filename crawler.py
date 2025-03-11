import sys
import asyncio
import aiohttp
import csv
import json
import os
import re
import logging
import argparse
import time
import platform
import random
import urllib3
import ctypes
from datetime import datetime
from urllib.parse import urlparse, urljoin, urlunparse
from bs4 import BeautifulSoup, Comment
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
import aiofiles
import hashlib
from collections import defaultdict
from typing import List, Tuple, Optional, Set, Dict, Any, Union
import subprocess
import yaml  # New import for YAML configuration
import gc  # Add garbage collection module

# Suppress InsecureRequestWarning messages
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Use SelectorEventLoop on Windows for compatibility with aiodns
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Global variables that will be loaded from config
USER_AGENTS = []
VIEWPORTS = []
CONCURRENT_WORKERS = None
REQUEST_TIMEOUT = None
CONN_LIMIT = None
MAX_RETRIES = None
MAX_CRAWL_DEPTH = None
MAX_URLS_PER_DOMAIN = None
RATE_LIMIT_PER_SECOND = None
MAX_REDIRECT_COUNT = None
RESPECT_ROBOTS_TXT = None
LOGS_DIR = None
CACHE_DIR = None

# Configure base logger
logger = logging.getLogger(__name__)

# Check if we're on Windows with Python 3.12 (problematic for Playwright)
PLAYWRIGHT_SUPPORTED = not (sys.platform.startswith("win") and sys.version_info.major == 3 and sys.version_info.minor >= 12)

# Check if curl is available
CURL_AVAILABLE = True  # Will be updated in main()

# Global persistent aiohttp session (created once for reuse)
aiohttp_session: Optional[aiohttp.ClientSession] = None

# Reuse browser instance
browser = None
domain_tracker = None

# Store robots.txt disallowed paths
robots_txt_cache = {}

def setup_directories_and_logging():
    """Set up directories and logging based on configured values"""
    # Create directories if they don't exist
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Configure logging to file and console
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"{LOGS_DIR}/scraper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", encoding="utf-8"),
            logging.StreamHandler()
        ]
    )

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    global USER_AGENTS, VIEWPORTS, CONCURRENT_WORKERS, REQUEST_TIMEOUT, CONN_LIMIT
    global MAX_RETRIES, MAX_CRAWL_DEPTH, MAX_URLS_PER_DOMAIN, RATE_LIMIT_PER_SECOND
    global MAX_REDIRECT_COUNT, RESPECT_ROBOTS_TXT, LOGS_DIR, CACHE_DIR

    default_config = {
        "crawler": {
            "max_depth": 5,  
            "max_urls_per_domain": 5000,
            "rate_limit": 20,
            "concurrent_workers": 10,
            "request_timeout": 30,
            "respect_robots_txt": False,
            "max_redirect_count": 5,
            "conn_limit": 50,
            "max_retries": 3,
        },
        "files": {
            "input_csv": "input.csv",
            "logs_dir": "DevCrawler-logs",  
            "cache_dir": "DevCrawler-cache",  
        },
        "browser": {
            "user_agents": [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36",
                "Mozilla/5.0 (iPad; CPU OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
                "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36 Edg/116.0.1938.81",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36 Brave/9.0.0.0",
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
            ],
            "viewports": [
                {'width': 1920, 'height': 1080},
                {'width': 1440, 'height': 900},
                {'width': 1366, 'height': 768},
                {'width': 1536, 'height': 864},
                {'width': 1280, 'height': 720},
                {'width': 360, 'height': 740},  # Mobile viewport
                {'width': 414, 'height': 896},  # iPhone viewport
            ]
        }
    }

    config = default_config
    
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                
            # Merge user config with default config
            if user_config:
                if 'crawler' in user_config:
                    config['crawler'].update(user_config['crawler'])
                if 'files' in user_config:
                    config['files'].update(user_config['files'])
                if 'browser' in user_config:
                    if 'user_agents' in user_config['browser']:
                        config['browser']['user_agents'] = user_config['browser']['user_agents']
                    if 'viewports' in user_config['browser']:
                        config['browser']['viewports'] = user_config['browser']['viewports']
        else:
            # If config doesn't exist, create it with default values
            with open(config_path, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False)
            print(f"Created default configuration file: {config_path}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        # Continue with default values
    
    # Load values from config
    USER_AGENTS = config['browser']['user_agents']
    VIEWPORTS = config['browser']['viewports']
    CONCURRENT_WORKERS = config['crawler']['concurrent_workers']
    REQUEST_TIMEOUT = config['crawler']['request_timeout']
    CONN_LIMIT = config['crawler']['conn_limit']
    MAX_RETRIES = config['crawler']['max_retries']
    MAX_CRAWL_DEPTH = config['crawler']['max_depth']
    MAX_URLS_PER_DOMAIN = config['crawler']['max_urls_per_domain']
    RATE_LIMIT_PER_SECOND = config['crawler']['rate_limit']
    MAX_REDIRECT_COUNT = config['crawler']['max_redirect_count']
    RESPECT_ROBOTS_TXT = config['crawler']['respect_robots_txt']
    LOGS_DIR = config['files']['logs_dir']
    CACHE_DIR = config['files']['cache_dir']
    
    return config

class DomainRateLimiter:
    """Rate limiter to prevent hammering servers"""
    def __init__(self, requests_per_second=2):
        self.delays = defaultdict(lambda: 1.0 / requests_per_second)
        self.last_request = defaultdict(float)
    
    async def wait(self, url: str):
        domain = get_base_domain(url)
        now = time.time()
        delay = self.delays[domain]
        last = self.last_request[domain]
        if now - last < delay:
            await asyncio.sleep(delay - (now - last))
        self.last_request[domain] = time.time()

class DomainFailureTracker:
    """Track failures for domains to avoid hammering failing servers"""
    def __init__(self):
        self.failures = defaultdict(int)
        self.max_failures = 3
        self.domains = set()
        self.url_counts = defaultdict(int)
        self.max_urls_per_domain = 50
        self.failed_urls = set()
        self.redirects = {}
        self.curl_failed_domains = set()
        self.failed_base_domains = set()
        self.external_links = []  # Store external links
    
    def add_domain(self, domain):
        """Add a domain to track"""
        self.domains.add(domain)
    
    def add_url(self, url):
        """Increment the URL count for a domain"""
        domain = get_base_domain(url)
        self.url_counts[domain] += 1
    
    def should_skip_url(self, url):
        """Check if we should skip this URL based on domain limits"""
        domain = get_base_domain(url)
        return self.url_counts[domain] >= self.max_urls_per_domain
    
    def record_failure(self, domain):
        """Record a failure for a domain"""
        self.failures[domain] += 1
    
    def mark_url_failed(self, url):
        """Mark a URL as failed"""
        self.failed_urls.add(url)
    
    def is_failed_url(self, url):
        """Check if a URL has been marked as failed"""
        return url in self.failed_urls
    
    def mark_curl_failed(self, url):
        """Mark a domain as having failed with curl"""
        domain = get_base_domain(url)
        self.curl_failed_domains.add(domain)
    
    def should_skip_curl(self, url):
        """Check if curl should be skipped for this domain"""
        domain = get_base_domain(url)
        return domain in self.curl_failed_domains
    
    def mark_base_url_failed(self, url):
        """Mark a base domain as having failed completely"""
        domain = get_base_domain(url)
        self.failed_base_domains.add(domain)
    
    def should_skip_domain(self, domain):
        """Check if a domain has too many failures"""
        return self.failures[domain] >= self.max_failures or domain in self.failed_base_domains
    
    def add_redirect(self, from_url, to_url):
        """Track redirects"""
        self.redirects[from_url] = to_url
        
    def add_external_link(self, source_url, external_url):
        """Store an external link"""
        self.external_links.append({
            'source': source_url,
            'external_url': external_url
        })

class UrlQueue:
    """Queue to manage URLs to be crawled with built-in deduplication"""
    def __init__(self):
        self.queue = []
        self.visited = set()
        self.in_progress = set()
        
    def add(self, url: str, depth: int = 0):
        if url not in self.visited and url not in self.in_progress and url not in [u for u, _ in self.queue]:
            self.queue.append((url, depth))
            
    def add_many(self, urls: List[str], depth: int = 0):
        for url in urls:
            self.add(url, depth)
            
    def get(self) -> Optional[Tuple[str, int]]:
        if not self.queue:
            return None
        url, depth = self.queue.pop(0)
        self.in_progress.add(url)
        return url, depth
    
    def mark_done(self, url: str):
        self.visited.add(url)
        if url in self.in_progress:
            self.in_progress.remove(url)
            
    def is_empty(self) -> bool:
        return len(self.queue) == 0
    
    def size(self) -> int:
        return len(self.queue)
    
    def visited_count(self) -> int:
        return len(self.visited)

def get_random_user_agent():
    """Return a random user agent from the list"""
    return random.choice(USER_AGENTS)

def get_random_viewport():
    """Return a random viewport from the list"""
    return random.choice(VIEWPORTS)

def sanitize_filename(s: str) -> str:
    """Convert URL to safe filename"""
    return re.sub(r"[^a-zA-Z0-9_\-\.]", "_", s)

def generate_filename(url: str) -> str:
    """Generate an appropriate filename from URL"""
    parsed = urlparse(url)
    path = parsed.path
    if not path or path == "/":
        return "index.html"
    path = path.strip("/")
    if "about" in path.lower():
        return "aboutus.html"
    filename = path.split("/")[-1]
    if not filename:
        # If the path ends with a slash, use the last directory name
        parts = path.split("/")
        for part in reversed(parts):
            if part:
                filename = part
                break
        if not filename:
            filename = "page"
    
    if not filename.endswith(".html"):
        filename += ".html"
    return sanitize_filename(filename)

def is_valid_page_url(url: str) -> bool:
    """Check if URL is suitable for crawling (exclude media, etc.)"""
    parsed = urlparse(url)
    ext = os.path.splitext(parsed.path)[1].lower()
    
    # Skip files with these extensions
    skip_extensions = [
        '.pdf', '.jpg', '.jpeg', '.png', '.gif', '.svg', '.ico', '.css', 
        '.js', '.mp3', '.mp4', '.mov', '.avi', '.zip', '.rar', '.tar',
        '.gz', '.7z', '.xml', '.json', '.rss', '.atom', '.doc', '.docx',
        '.xls', '.xlsx', '.ppt', '.pptx'
    ]
    
    if ext in skip_extensions:
        return False
        
    # Skip URLs with these parameters or patterns
    lower = url.lower()
    if any(x in lower for x in [
        "format=xml", "format=json", "feed=", "rss", "atom", "print=", 
        "download=", "attachment", "calendar", ".ashx", ".axd", 
        "/wp-admin/", "/wp-login", "/wp-content/uploads/",
        "/admin/", "/login", "/logout", "/signin", "/signout",
        "share=", "comment", "/search/"
    ]):
        return False
        
    return True

def get_base_domain(url: str) -> str:
    """Extract base domain from URL"""
    return urlparse(url).netloc

def get_base_url(url: str) -> str:
    """Get base URL with scheme and domain but no path"""
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"

def is_internal(link: str, base_domain: str) -> bool:
    """Check if link is internal to the site"""
    link_domain = urlparse(link).netloc
    
    # Handle 'www' variations
    base_parts = base_domain.split('.')
    link_parts = link_domain.split('.')
    
    # Remove 'www' if present
    if base_parts[0] == 'www':
        base_parts = base_parts[1:]
    if link_parts[0] == 'www':
        link_parts = link_parts[1:]
        
    # Compare domains without 'www'
    return '.'.join(base_parts) == '.'.join(link_parts)

def normalize_domain(url: str) -> str:
    """Normalize domain by handling www variations"""
    parsed = urlparse(url)
    domain = parsed.netloc
    
    # Standardize domain with or without www.
    parts = domain.split('.')
    if parts[0] == 'www':
        standardized = '.'.join(parts[1:])
    else:
        standardized = domain
        
    # Reconstruct URL with normalized domain
    new_parsed = parsed._replace(netloc=standardized)
    return urlunparse(new_parsed)

def extract_links(soup: BeautifulSoup, base_url: str, base_domain: str = None) -> List[str]:
    """Extract and normalize links from a BeautifulSoup object"""
    if base_domain is None:
        base_domain = get_base_domain(base_url)
        
    links = []
    external_links = []
    
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        
        # Skip empty links, javascript, mailto, tel links
        if not href or href.startswith(('javascript:', 'mailto:', 'tel:')):
            continue
            
        # Normalize URL
        full_url = urljoin(base_url, href)
        
        # Skip fragments within the same page
        if full_url.split('#')[0] == base_url.split('#')[0]:
            continue
            
        # Check if it's an internal or external link
        link_domain = get_base_domain(full_url)
        if is_internal(full_url, base_domain):
            # Internal link
            normalized_url = normalize_url(full_url)
            if normalized_url:
                links.append(normalized_url)
        else:
            # External link - track it
            if domain_tracker is not None:
                domain_tracker.add_external_link(base_url, full_url)
            external_links.append(full_url)
            
    return list(set(links))  # Remove duplicates

def save_html(domain: str, url: str, html: str):
    """Save HTML content to disk immediately and clear memory"""
    domain_dir = os.path.join(LOGS_DIR, domain)
    os.makedirs(domain_dir, exist_ok=True)
    filename = generate_filename(url)
    filepath = os.path.join(domain_dir, filename)
    try:
        # Use errors="replace" to handle encoding issues
        with open(filepath, "w", encoding="utf-8", errors="replace") as f:
            f.write(html)
        logger.debug(f"Saved HTML for {url} to {filepath}")
        # Clear html from memory
        html = None
        gc.collect(0)  # Run a quick generation 0 garbage collection
    except Exception as e:
        logger.error(f"Error saving HTML for {url}: {e}")

async def stealth_page_config(page):
    """Configure Playwright page to avoid detection as bot"""
    await page.add_init_script("""
        Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
        Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
        Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3] });
        window.chrome = { runtime: {} };
        const originalQuery = window.navigator.permissions.query;
        window.navigator.permissions.query = (parameters) =>
          parameters.name === 'notifications' ?
          Promise.resolve({ state: Notification.permission }) :
          originalQuery(parameters);
    """)

def handle_async_exceptions(task):
    """Handle exceptions in async tasks to prevent 'Task exception never retrieved' errors."""
    try:
        # Check if the task has an exception
        if task.exception():
            logger.error(f"Async task error: {task.exception()}")
    except (asyncio.CancelledError, asyncio.InvalidStateError):
        # Task was cancelled or is not done yet
        pass

def check_curl_available():
    """Check if curl is available on the system."""
    try:
        result = subprocess.run(['curl', '--version'], 
                               capture_output=True, 
                               text=True, 
                               timeout=5)
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def get_cloudflare_headers(referrer=None):
    """
    Generate realistic browser headers that help bypass Cloudflare protection.
    Uses modern browser fingerprinting resistance techniques.
    """
    # Get a random user agent
    user_agent = get_random_user_agent()
    
    # Choose a random browser for sec-ch-ua
    browsers = [
        '"Chromium";v="116", "Not:A-Brand";v="24", "Google Chrome";v="116"',
        '"Chromium";v="118", "Not:A-Brand";v="99", "Google Chrome";v="118"',
        '"Chromium";v="122", "Not:A-Brand";v="8", "Google Chrome";v="122"',
        '"Chromium";v="134", "Not:A-Brand";v="24", "Brave";v="134"',
        '"Microsoft Edge";v="118", "Chromium";v="118", "Not:A-Brand";v="99"',
        '"Not.A/Brand";v="8", "Chromium";v="114", "Safari";v="114"'
    ]
    sec_ch_ua = random.choice(browsers)
    
    # Choose a random platform
    platforms = ['"Windows"', '"macOS"', '"Linux"', '"Android"']
    platform = random.choice(platforms)
    
    # Choose a random mobile value
    mobile_values = ["?0", "?1"]
    is_mobile = mobile_values[0] if platform in ['"Windows"', '"macOS"', '"Linux"'] else mobile_values[1]
    
    # Choose a random language
    languages = [
        "en-US,en;q=0.9",
        "en-GB,en;q=0.9",
        "en-IN,en;q=0.7",
        "en-CA,en;q=0.9,fr-CA;q=0.8",
        "es-ES,es;q=0.9,en;q=0.8",
        "fr-FR,fr;q=0.9,en;q=0.8",
        "de-DE,de;q=0.9,en;q=0.8"
    ]
    accept_language = random.choice(languages)
    
    # Choose random fetch metadata
    fetch_dests = ["empty", "document", "image"]
    fetch_modes = ["cors", "navigate", "no-cors"]
    fetch_sites = ["same-origin", "same-site", "cross-site"]
    
    # Set referer if provided
    referer_header = {}
    if referrer:
        referer_header = {"Referer": referrer}
    
    # Build the headers
    headers = {
        "User-Agent": user_agent,
        "Accept": "*/*",
        "Accept-Language": accept_language,
        "Accept-Encoding": "gzip, deflate, br",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Connection": "keep-alive",
        "sec-ch-ua": sec_ch_ua,
        "sec-ch-ua-mobile": is_mobile,
        "sec-ch-ua-platform": platform,
        "sec-fetch-dest": random.choice(fetch_dests),
        "sec-fetch-mode": random.choice(fetch_modes),
        "sec-fetch-site": random.choice(fetch_sites),
        "DNT": "1",  # Do Not Track
        "Upgrade-Insecure-Requests": "1",
        **referer_header
    }
    
    # Randomly add Brave's Global Privacy Control
    if random.random() < 0.3:  # 30% chance to add GPC
        headers["sec-gpc"] = "1"
    
    return headers

def is_cloudflare_site(url: str) -> bool:
    """
    Enhanced detection of Cloudflare-protected sites.
    Checks domain and also looks for common Cloudflare patterns in content.
    """
    cloudflare_domains = ["cloudflare.com", "workers.dev", "pages.dev"]
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    
    # Check if it's a direct Cloudflare domain
    if any(cf_domain in domain for cf_domain in cloudflare_domains):
        return True
    
    # Try to check cached content for Cloudflare signatures
    cache_key = hashlib.md5(url.encode()).hexdigest()
    cache_file = f"{CACHE_DIR}/{cache_key}.html"
    
    try:
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read(10000)  # Read first 10KB to check
                
                # Look for Cloudflare signatures in the content
                cf_patterns = [
                    "cloudflare", "cf-ray", "cf-browser-verification",
                    "cf_chl_", "cf-please-wait", "__cf_", "turnstile"
                ]
                
                return any(pattern in content.lower() for pattern in cf_patterns)
    except:
        pass
    
    return False

def check_robots_txt(domain):
    """Simple function to check if robots.txt exists and basic parsing"""
    if domain in robots_txt_cache:
        return robots_txt_cache[domain]
        
    robots_url = f"https://{domain}/robots.txt"
    try:
        import requests
        response = requests.get(robots_url, timeout=10, verify=False)
        if response.status_code == 200:
            lines = response.text.splitlines()
            disallowed = []
            for line in lines:
                if line.lower().startswith('disallow:'):
                    path = line.split(':', 1)[1].strip()
                    if path:
                        disallowed.append(path)
            logger.info(f"Found {len(disallowed)} disallowed paths in robots.txt for {domain}")
            robots_txt_cache[domain] = disallowed
            return disallowed
        robots_txt_cache[domain] = []
        return []
    except Exception:
        robots_txt_cache[domain] = []
        return []

def should_respect_robots(url, disallowed_paths):
    """Check if URL matches any disallowed paths in robots.txt"""
    if not disallowed_paths:
        return False
        
    parsed = urlparse(url)
    path = parsed.path
    
    for disallowed in disallowed_paths:
        if disallowed == '/':  # Complete disallow
            return True
        if path.startswith(disallowed):
            return True
    return False

async def get_browser():
    """Get or create the global browser instance with better error handling."""
    global browser
    
    # Skip browser initialization for Python 3.12 on Windows
    if not PLAYWRIGHT_SUPPORTED:
        logger.debug("Playwright not supported on this system configuration")
        return None
        
    if browser is None:
        try:
            # Wrap the entire Playwright initialization in a try-except
            try:
                p = await async_playwright().start()
                browser = await p.chromium.launch(
                    headless=True,
                    args=[
                        '--no-sandbox', 
                        '--disable-dev-shm-usage',
                        '--disable-features=IsolateOrigins,site-per-process',
                        '--disable-site-isolation-trials'
                    ]
                )
            except NotImplementedError:
                logger.warning("Playwright subprocess creation failed with NotImplementedError. "
                              "This is a known issue with Python 3.12 on Windows. "
                              "Falling back to curl-only mode.")
                return None
            except Exception as e:
                logger.warning(f"Playwright initialization failed: {e}. Falling back to curl-only mode.")
                return None
        except Exception as e:
            logger.warning(f"Unexpected error initializing Playwright: {e}")
            return None
    return browser

async def get_aiohttp_session() -> aiohttp.ClientSession:
    """Get or create the global aiohttp session"""
    global aiohttp_session
    if aiohttp_session is None or aiohttp_session.closed:
        connector = aiohttp.TCPConnector(limit=CONN_LIMIT, ttl_dns_cache=300)
        timeout = aiohttp.ClientTimeout(
            total=REQUEST_TIMEOUT,
            connect=10,
            sock_connect=10,
            sock_read=REQUEST_TIMEOUT
        )
        aiohttp_session = aiohttp.ClientSession(
            connector=connector,
            headers={'User-Agent': get_random_user_agent()},
            timeout=timeout
        )
    return aiohttp_session

def try_other_url_variants(url: str) -> List[str]:
    """Generate URL variants to try if the original fails"""
    parsed = urlparse(url)
    variants = []
    
    # Try with/without www
    if parsed.netloc.startswith('www.'):
        new_netloc = parsed.netloc[4:]  # Remove www.
        variants.append(urlunparse(parsed._replace(netloc=new_netloc)))
    else:
        new_netloc = 'www.' + parsed.netloc  # Add www.
        variants.append(urlunparse(parsed._replace(netloc=new_netloc)))
    
    # Try with/without trailing slash
    if parsed.path == '':
        variants.append(urlunparse(parsed._replace(path='/')))
    elif parsed.path == '/':
        variants.append(urlunparse(parsed._replace(path='')))
        
    # Try both http and https
    http_variant = urlunparse(parsed._replace(scheme='http'))
    https_variant = urlunparse(parsed._replace(scheme='https'))
    
    if parsed.scheme == 'https':
        variants.append(http_variant)
    else:
        variants.append(https_variant)
    
    return variants

async def fetch_with_curl(url: str, max_redirects=MAX_REDIRECT_COUNT) -> Optional[Tuple[str, str]]:
    """Fetch URL using aiohttp (named curl for historical reasons)"""
    global domain_tracker
    
    if domain_tracker.is_failed_url(url):
        logger.debug(f"Skipping previously failed URL: {url}")
        return None
        
    if domain_tracker.should_skip_curl(url):
        logger.debug(f"Skipping curl for known failed domain: {get_base_domain(url)}")
        return None

    try:
        session = await get_aiohttp_session()
        # Use a random user agent for each request
        headers = {'User-Agent': get_random_user_agent()}
        
        async with session.get(
            url, 
            allow_redirects=True,
            max_redirects=max_redirects,
            verify_ssl=False,  # Allow self-signed certs
            headers=headers
        ) as resp:
            if resp.status >= 400:
                logger.warning(f"HTTP error {resp.status} for {url}")
                
                # Try other variants if we get a 404
                if resp.status == 404:
                    for variant in try_other_url_variants(url):
                        logger.info(f"Trying URL variant: {variant}")
                        result = await fetch_with_curl(variant, max_redirects=1)
                        if result:
                            return result
                
                domain_tracker.mark_url_failed(url)
                return None
                
            # Check for too many redirects
            if len(resp.history) >= max_redirects:
                logger.warning(f"Too many redirects for {url}")
                domain_tracker.mark_url_failed(url)
                return None
                
            # Successful fetch
            final_url = str(resp.url)
            
            # Track redirect
            if final_url != url:
                domain_tracker.add_redirect(url, final_url)
                
            try:
                text = await resp.text()
                return (final_url, text)
            except UnicodeDecodeError:
                # Try binary read for encoding issues
                binary = await resp.read()
                try:
                    # Try utf-8 with error replacement
                    text = binary.decode('utf-8', errors='replace')
                    return (final_url, text)
                except Exception as e:
                    logger.error(f"Binary decode error for {url}: {e}")
                    domain_tracker.mark_curl_failed(url)
    except aiohttp.ClientError as e:
        logger.error(f"Curl client error for {url}: {e}")
        domain_tracker.mark_curl_failed(url)
    except asyncio.TimeoutError:
        logger.error(f"Curl timeout error for {url}")
        domain_tracker.mark_curl_failed(url)
    except Exception as e:
        logger.error(f"Curl fetch error for {url}: {e}")
        domain_tracker.mark_curl_failed(url)
        domain_tracker.mark_base_url_failed(url)
    
    return None

async def fetch_with_playwright(url: str, max_redirects=MAX_REDIRECT_COUNT) -> Optional[Tuple[str, str]]:
    """Fetch URL using Playwright (for JavaScript-heavy sites)"""
    # Skip Playwright if we're on Windows with Python 3.12
    if not PLAYWRIGHT_SUPPORTED:
        logger.debug(f"Playwright not supported on this system. Skipping for {url}")
        return None
        
    try:
        browser = await get_browser()
        if browser is None:
            logger.debug(f"Skipping Playwright fetch for {url} as browser initialization failed")
            return None
            
        # Use random user agent and viewport
        random_user_agent = get_random_user_agent()
        random_viewport = get_random_viewport()
            
        context = await browser.new_context(
            user_agent=random_user_agent,
            viewport=random_viewport,
            java_script_enabled=True,
            ignore_https_errors=True  # Allow self-signed certs
        )
        page = await context.new_page()
        await stealth_page_config(page)
        
        # Set timeout longer than the default
        page.set_default_timeout(REQUEST_TIMEOUT * 1000)
        
        # Handle response to detect status codes
        response_status = 0
        
        def handle_response(response):
            nonlocal response_status
            if response.url == url:
                response_status = response.status
        
        page.on("response", handle_response)
        
        try:
            # Navigate with timeout and wait for network idle
            response = await page.goto(
                url, 
                wait_until="networkidle", 
                timeout=REQUEST_TIMEOUT * 1000
            )
            
            # Check status code
            if response and response.status >= 400:
                logger.warning(f"Playwright HTTP error {response.status} for {url}")
                domain_tracker.mark_url_failed(url)
                await context.close()
                return None
                
        except PlaywrightTimeoutError:
            # If we timeout waiting for networkidle, try to get content anyway
            logger.warning(f"Playwright timeout waiting for networkidle on {url}")
            
        # Check for redirect chain length
        redirects = await page.evaluate("""() => {
            return window.performance
                .getEntriesByType('navigation')
                .map(nav => nav.redirectCount)[0] || 0
        }""")
        
        if redirects >= max_redirects:
            logger.warning(f"Too many redirects ({redirects}) for {url}")
            domain_tracker.mark_url_failed(url)
            await context.close()
            return None
            
        # Get final URL and content
        final_url = page.url
        if final_url != url:
            domain_tracker.add_redirect(url, final_url)
            
        content = await page.content()
        await context.close()
        
        # Skip empty responses
        if not content or len(content.strip()) < 100:
            logger.warning(f"Empty or too small content from {url}")
            return None
            
        return (final_url, content)
        
    except Exception as e:
        logger.error(f"Playwright error for {url}: {e}")
        domain_tracker.mark_url_failed(url)
    
    return None

async def crawl_with_curl(url):
    """Crawl a website using curl as a fallback with better encoding handling."""
    try:
        # For Windows with encoding issues
        if sys.platform.startswith("win"):
            # Try to set console to UTF-8 mode (helps with some encoding issues)
            try:
                if hasattr(ctypes, 'windll'):
                    kernel32 = ctypes.windll.kernel32
                    kernel32.SetConsoleCP(65001)  # Set console input to UTF-8
                    kernel32.SetConsoleOutputCP(65001)  # Set console output to UTF-8
            except Exception:
                pass  # Ignore if this fails
                
            # Use direct binary mode and handle encoding ourselves
            command = ['curl', '-L', '-A', get_random_user_agent(), url]
            try:
                # Use binary mode to avoid Windows console encoding issues
                result = subprocess.run(
                    command, 
                    capture_output=True,  # Binary output
                    check=False,
                    timeout=30
                )
                
                if result.returncode == 0:
                    # Try to decode with multiple encodings
                    for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                        try:
                            return result.stdout.decode(encoding, errors='replace')
                        except Exception:
                            continue
                    
                    # Last resort - force decode as latin-1 (which never fails)
                    return result.stdout.decode('latin-1', errors='replace')
                else:
                    logger.error(f"Curl error: {result.stderr.decode('utf-8', errors='replace')}")
                    return None
            except subprocess.TimeoutExpired:
                logger.error(f"Curl timeout for {url}")
                return None
            except Exception as e:
                logger.error(f"Error using curl with binary mode: {e}")
                return None
        else:
            # For non-Windows platforms, use the original implementation
            command = ['curl', '-L', '-A', get_random_user_agent(), url]
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                return result.stdout
            else:
                logger.error(f"Curl error: {result.stderr}")
                return None
    except Exception as e:
        logger.error(f"Error using curl: {e}")
        return None

async def fetch_with_powershell(url):
    """Try to fetch using PowerShell's Invoke-WebRequest (Windows only)"""
    if not sys.platform.startswith("win"):
        return None
        
    try:
        user_agent = get_random_user_agent()
        # PowerShell command with error handling
        ps_command = f'powershell -Command "$ProgressPreference = \'SilentlyContinue\'; try {{ Invoke-WebRequest -Uri \'{url}\' -UseBasicParsing -UserAgent \'{user_agent}\' | Select-Object -ExpandProperty Content }} catch {{ $_.Exception.Message }}"'
        
        # Run PowerShell command
        result = subprocess.run(
            ps_command, 
            capture_output=True, 
            shell=True,
            timeout=30
        )
        
        if result.returncode == 0 and len(result.stdout) > 200:
            return result.stdout.decode('utf-8', errors='replace')
        else:
            error = result.stderr.decode('utf-8', errors='replace')
            if error:
                logger.debug(f"PowerShell fetch error: {error}")
            return None
    except Exception as e:
        logger.debug(f"PowerShell fetch failed: {e}")
        return None

async def fetch_url_cloudflare(url: str, rate_limiter) -> Optional[Tuple[str, str, List[str]]]:
    """Enhanced version for fetching from Cloudflare-protected sites"""
    global domain_tracker
    
    if domain_tracker.is_failed_url(url):
        return None
    
    await rate_limiter.wait(url)
    
    # Get the base domain to use as the referrer
    parsed = urlparse(url)
    base_url = f"{parsed.scheme}://{parsed.netloc}"
    
    # Use our more sophisticated headers
    headers = get_cloudflare_headers(referrer=base_url)
    
    # Attempt with aiohttp but with improved headers
    session = await get_aiohttp_session()
    try:
        async with session.get(
            url,
            allow_redirects=True,
            max_redirects=MAX_REDIRECT_COUNT,
            verify_ssl=False,
            headers=headers
        ) as resp:
            if resp.status >= 400:
                logger.warning(f"HTTP error {resp.status} for Cloudflare site {url}")
                return None
            
            final_url = str(resp.url)
            text = await resp.text()
            await save_to_cache(url, text)
            links = extract_links(BeautifulSoup(text, "html.parser"), final_url, get_base_domain(final_url))
            return final_url, text, links
    except Exception as e:
        logger.warning(f"aiohttp error for Cloudflare site {url}: {e}")
    
    # If aiohttp fails, try with requests library
    try:
        import requests
        
        # Add a small sleep to mimic human behavior
        await asyncio.sleep(random.uniform(1, 3))
        
        # Create a session with cookiejar support
        session = requests.Session()
        
        # Make the request
        response = session.get(
            url,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
            verify=False,
            allow_redirects=True
        )
        
        if response.status_code < 400:
            html = response.text
            await save_to_cache(url, html)
            links = extract_links(BeautifulSoup(html, "html.parser"), response.url, get_base_domain(response.url))
            return response.url, html, links
    except Exception as e:
        logger.error(f"Requests error for Cloudflare site {url}: {e}")
    
    return None

async def fetch_with_requests(url: str, max_redirects=5) -> Optional[Tuple[str, str]]:
    """Fetch URL using the requests library as a last resort fallback."""
    try:
        import requests
        headers = {'User-Agent': get_random_user_agent()}
        response = requests.get(
            url,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
            verify=False,  # Allow self-signed certs
            allow_redirects=True
        )
        
        if response.status_code >= 400:
            logger.warning(f"HTTP error {response.status_code} for {url}")
            return None
            
        return (response.url, response.text)
    except Exception as e:
        logger.error(f"Requests error for {url}: {e}")
        return None

async def get_cached_content(url: str) -> Optional[Tuple[str, str]]:
    """Get cached content for URL if available"""
    cache_key = hashlib.md5(url.encode()).hexdigest()
    cache_file = f"{CACHE_DIR}/{cache_key}.html"
    try:
        if os.path.exists(cache_file):
            async with aiofiles.open(cache_file, mode='r', encoding="utf-8", errors="replace") as f:
                content = await f.read()
                return (url, content)
    except Exception:
        pass
    return None

async def save_to_cache(url: str, content: str):
    """Save content to cache immediately and clear memory"""
    cache_key = hashlib.md5(url.encode()).hexdigest()
    cache_file = f"{CACHE_DIR}/{cache_key}.html"
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        async with aiofiles.open(cache_file, mode='w', encoding="utf-8", errors="replace") as f:
            await f.write(content)
        # No need to keep content in memory after saving
        content = None
    except Exception as e:
        logger.error(f"Cache save error for {url}: {e}")

def convert_html_to_markdown(html: str, url: str) -> str:
    """
    Convert HTML content to text format with minimal assumptions about structure.
    Works with websites of all types, languages, and character sets.
    """
    try:
        # Create soup
        soup = BeautifulSoup(html, "html.parser")
        
        # Remove comments
        for comment in soup.find_all(text=lambda text: isinstance(text, Comment)):
            comment.extract()
        
        # Extract title and description if present (don't assume they exist)
        title_text = "No Title"
        if soup.title and soup.title.string:
            title_text = soup.title.string.strip()
        
        meta_desc = ""
        meta_tag = soup.find("meta", attrs={"name": ["description", "Description"]})
        if meta_tag and meta_tag.get("content"):
            meta_desc = meta_tag.get("content").strip()
        
        # Remove unwanted tags that typically don't contain meaningful content
        for tag in soup.find_all(['script', 'style', 'noscript', 'svg', 'path', 'symbol', 
                                  'defs', 'rect', 'circle', 'polygon', 'linearGradient']):
            tag.decompose()
        
        # Extract text content
        text = ""
        
        # First attempt: Try to find main content area if it exists
        main_content_candidates = soup.find_all(['main', 'article', 'div', 'section'], 
                                              id=re.compile('(content|main|article|post)', re.I))
        main_content_candidates += soup.find_all(['main', 'article', 'div', 'section'], 
                                             class_=re.compile('(content|main|article|post)', re.I))
        
        # If we found potential main content areas, use them
        if main_content_candidates:
            for content_area in main_content_candidates:
                # Get all text, removing excessive whitespace
                for element in content_area.find_all(text=True):
                    if element.parent.name not in ['script', 'style', 'meta', 'link']:
                        s = element.strip()
                        if s:
                            text += s + " "
        else:
            # If no main content found, extract from body with some structure preservation
            if soup.body:
                # Process all text nodes in body
                for element in soup.body.find_all(text=True, recursive=True):
                    if element.parent.name not in ['script', 'style', 'meta', 'link']:
                        s = element.strip()
                        if s:
                            # Add newlines for block elements to preserve some structure
                            if element.parent.name in ['p', 'div', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 
                                                      'header', 'footer', 'blockquote', 'pre', 'br', 'hr',
                                                      'table', 'tr', 'section', 'article']:
                                text += s + "\n\n"
                            else:
                                text += s + " "
            else:
                # Fallback: just get all text from the document
                for element in soup.find_all(text=True, recursive=True):
                    if element.parent.name not in ['script', 'style', 'meta', 'link']:
                        s = element.strip()
                        if s:
                            text += s + " "
        
        # Clean up text - normalize whitespace
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Split into paragraphs for readability
        paragraphs = []
        current_paragraph = ""
        
        for line in text.split('\n'):
            line = line.strip()
            if line:
                current_paragraph += line + " "
            elif current_paragraph:
                paragraphs.append(current_paragraph.strip())
                current_paragraph = ""
        
        if current_paragraph:
            paragraphs.append(current_paragraph.strip())
        
        # Join paragraphs with double newlines
        clean_text = "\n\n".join(paragraphs)
        
        # Handle empty content
        if not clean_text.strip():
            clean_text = "No content extracted from page"
        
        # Format as markdown with minimal structure
        md = f"URL: {url}\n\n"
        if title_text:
            md += f"Page Title: {title_text}\n\n"
        if meta_desc:
            md += f"Meta Description: {meta_desc}\n\n"
        md += "---\n\n" + clean_text
        
        return md
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"URL: {url}\n\nError converting content: {str(e)}\n\nDetails: {error_details}"

def save_markdown(domain: str, url: str, markdown_text: str) -> str:
    """Save markdown content to disk immediately, clear memory and return the filepath"""
    domain_dir = os.path.join(LOGS_DIR, domain)
    os.makedirs(domain_dir, exist_ok=True)
    filename = generate_filename(url)
    filename = os.path.splitext(filename)[0] + ".md"
    filepath = os.path.join(domain_dir, filename)
    try:
        with open(filepath, "w", encoding="utf-8", errors="replace") as f:
            f.write(markdown_text)
        logger.debug(f"Saved markdown for {url} to {filepath}")
        # Clear markdown from memory
        markdown_text = None
        return filepath
    except Exception as e:
        logger.error(f"Error saving markdown for {url}: {e}")
        return ""

def normalize_url(url: str) -> str:
    """Ensure URL starts with http:// or https:// and normalize"""
    url = url.strip()
    
    # Add scheme if missing
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    
    # Parse and normalize
    parsed = urlparse(url)
    
    # Remove default ports
    if (parsed.port == 80 and parsed.scheme == 'http') or (parsed.port == 443 and parsed.scheme == 'https'):
        netloc = parsed.netloc.split(':')[0]
        parsed = parsed._replace(netloc=netloc)
    
    # Ensure path exists
    if not parsed.path:
        parsed = parsed._replace(path='/')
    
    # Remove trailing slash from path if it's just a single slash
    if parsed.path == '/' and not parsed.query and not parsed.fragment:
        pass  # Keep the single slash
    elif parsed.path.endswith('/') and len(parsed.path) > 1:
        parsed = parsed._replace(path=parsed.path.rstrip('/'))
    
    # Return normalized URL
    return urlunparse(parsed)

async def fetch_url(url: str, rate_limiter) -> Optional[Tuple[str, str, List[str]]]:
    """Fetch a URL and return the final URL, HTML content, and extracted links with memory optimization"""
    global domain_tracker, CURL_AVAILABLE
    
    if domain_tracker.is_failed_url(url):
        logger.debug(f"Skipping previously failed URL: {url}")
        return None
        
    # Check for robots.txt restrictions
    if RESPECT_ROBOTS_TXT:
        domain = get_base_domain(url)
        disallowed_paths = check_robots_txt(domain)
        if should_respect_robots(url, disallowed_paths):
            logger.info(f"Skipping URL {url} due to robots.txt restrictions")
            domain_tracker.mark_url_failed(url)
            return None
    
    # Special handling for Cloudflare sites
    is_cloudflare = is_cloudflare_site(url)
    if is_cloudflare:
        logger.info(f"Cloudflare site detected: {url} - using special handling")
        cloudflare_result = await fetch_url_cloudflare(url, rate_limiter)
        if cloudflare_result:
            return cloudflare_result
    
    # Check cache first
    cached_result = await get_cached_content(url)
    if cached_result:
        final_url, html = cached_result
        logger.info(f"Using cached content for {url}")
        links = extract_links(BeautifulSoup(html, "html.parser"), final_url, get_base_domain(final_url))
        return final_url, html, links

    # Try with aiohttp first (lightweight)
    if not domain_tracker.should_skip_curl(url):
        await rate_limiter.wait(url)
        
        # Normal fetch
        result = await fetch_with_curl(url)
        if result:
            final_url, html = result
            await save_to_cache(url, html)
            
            # Check for empty or too small responses
            if not html or len(html.strip()) < 200:
                logger.warning(f"Too small content from curl for {url}, trying other methods")
            else:
                links = extract_links(BeautifulSoup(html, "html.parser"), final_url, get_base_domain(final_url))
                return final_url, html, links
    
    # Fall back to Playwright if supported
    if PLAYWRIGHT_SUPPORTED:
        await rate_limiter.wait(url)
        playwright_result = await fetch_with_playwright(url)
        if playwright_result:
            final_url, html = playwright_result
            await save_to_cache(url, html)
            links = extract_links(BeautifulSoup(html, "html.parser"), final_url, get_base_domain(final_url))
            return final_url, html, links
    
    # Try command-line curl with improved encoding handling
    if CURL_AVAILABLE:
        logger.info(f"Trying command-line curl for {url}")
        await rate_limiter.wait(url)
        html_content = await crawl_with_curl(url)
        if html_content:
            await save_to_cache(url, html_content)
            links = extract_links(BeautifulSoup(html_content, "html.parser"), url, get_base_domain(url))
            return url, html_content, links
    
    # Try PowerShell on Windows systems
    if sys.platform.startswith("win"):
        logger.info(f"Trying PowerShell fetch for {url}")
        await rate_limiter.wait(url)
        html_content = await fetch_with_powershell(url)
        if html_content:
            await save_to_cache(url, html_content)
            links = extract_links(BeautifulSoup(html_content, "html.parser"), url, get_base_domain(url))
            return url, html_content, links
    
    # Last resort: try with requests library
    logger.info(f"Trying requests library for {url}")
    await rate_limiter.wait(url)
    requests_result = await fetch_with_requests(url)
    if requests_result:
        final_url, html = requests_result
        await save_to_cache(url, html)
        links = extract_links(BeautifulSoup(html, "html.parser"), final_url, get_base_domain(final_url))
        return final_url, html, links
    
    # If all methods failed, try URL variants
    variants = try_other_url_variants(url)
    for variant in variants:
        logger.info(f"Trying URL variant: {variant}")
        # Only try curl for variants to keep it faster
        await rate_limiter.wait(variant)
        result = await fetch_with_curl(variant, max_redirects=2)
        if result:
            final_url, html = result
            await save_to_cache(url, html)  # Cache with original URL
            domain_tracker.add_redirect(url, final_url)  # Track redirect
            links = extract_links(BeautifulSoup(html, "html.parser"), final_url, get_base_domain(final_url))
            return final_url, html, links
    
    # All attempts failed
    logger.warning(f"All fetch attempts failed for {url}")
    domain_tracker.mark_url_failed(url)
    # Force garbage collection to clear any partial results
    gc.collect(0)
    return None

def trim_repeated_content(text: str) -> str:
    """
    Heuristically remove duplicate or boilerplate lines from text.
    This function:
      - Splits the text into lines.
      - Counts the frequency of each normalized (stripped, lowercased) line.
      - Removes lines that are very short (<3 words) or that appear more than a threshold (e.g. > 3 times).
      - Also skips consecutive duplicate lines.
    """
    lines = text.splitlines()
    if not lines:
        return text

    # Count normalized line frequencies
    freq = {}
    for line in lines:
        norm = line.strip().lower()
        if norm:
            freq[norm] = freq.get(norm, 0) + 1

    cleaned_lines = []
    previous = None
    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue
        # Skip very short lines (less than 3 words)
        if len(line_clean.split()) < 3:
            continue
        # Skip lines that appear too frequently (threshold > 3)
        if freq.get(line_clean.lower(), 0) > 3:
            continue
        # Skip duplicate consecutive lines
        if line_clean == previous:
            continue
        cleaned_lines.append(line_clean)
        previous = line_clean
    return "\n".join(cleaned_lines)

def extract_url_from_md(md_text: str) -> str:
    """
    Extract the URL from the Markdown text.
    Assumes the first line that starts with 'URL:' contains the URL.
    """
    for line in md_text.splitlines():
        if line.strip().lower().startswith("url:"):
            return line.split(":", 1)[1].strip()
    return ""

def process_markdown_file(filepath: str) -> dict:
    """Extract structured data from a markdown file"""
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
            
        # Extract URL
        url = extract_url_from_md(content)
        
        # Extract title
        title = ""
        for line in content.splitlines():
            if line.strip().lower().startswith("page title:"):
                title = line.split(":", 1)[1].strip()
                break
                
        # Extract description
        description = ""
        for line in content.splitlines():
            if line.strip().lower().startswith("meta description:"):
                description = line.split(":", 1)[1].strip()
                break
                
        # Extract main content (after the "---" separator)
        main_content = ""
        content_started = False
        for line in content.splitlines():
            if content_started:
                main_content += line + "\n"
            elif line.strip() == "---":
                content_started = True
                
        # Clean up content
        main_content = trim_repeated_content(main_content)
        
        return {
            "url": url,
            "title": title,
            "description": description,
            "content": main_content
        }
    except Exception as e:
        logger.error(f"Error processing markdown file {filepath}: {e}")
        return {
            "url": "",
            "title": "",
            "description": "",
            "content": ""
        }

def save_external_links(domain: str):
    """Save external links to a CSV file"""
    global domain_tracker
    
    if not domain_tracker or not domain_tracker.external_links:
        return
        
    filepath = os.path.join(LOGS_DIR, f"{domain}_external_links.csv")
    try:
        with open(filepath, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Source URL", "External URL"])
            for link in domain_tracker.external_links:
                writer.writerow([link['source'], link['external_url']])
        logger.info(f"Saved {len(domain_tracker.external_links)} external links to {filepath}")
    except Exception as e:
        logger.error(f"Error saving external links: {e}")

async def crawl_domain(start_url: str, max_depth: int = 3, max_urls: int = 50, rate_limit: float = 2) -> Set[str]:
    """Crawl a domain starting from the given URL with memory optimization"""
    global domain_tracker, browser
    
    # Initialize domain tracker if not already done
    if domain_tracker is None:
        domain_tracker = DomainFailureTracker()
        
    # Initialize rate limiter
    rate_limiter = DomainRateLimiter(requests_per_second=rate_limit)
    
    # Normalize the start URL
    start_url = normalize_url(start_url)
    base_domain = get_base_domain(start_url)
    
    # Track this domain
    domain_tracker.add_domain(base_domain)
    
    # URL queue to manage URLs
    url_queue = UrlQueue()
    url_queue.add(start_url, 0)  # Add start URL with depth 0
    
    # Process URLs until queue is empty or max URLs reached
    while not url_queue.is_empty() and url_queue.visited_count() < max_urls:
        url_depth = url_queue.get()
        if not url_depth:
            break
            
        current_url, depth = url_depth
        
        # Skip if domain has too many failures
        if domain_tracker.should_skip_domain(get_base_domain(current_url)):
            url_queue.mark_done(current_url)
            continue
            
        # Skip if max URLs for this domain reached
        if domain_tracker.should_skip_url(current_url):
            url_queue.mark_done(current_url)
            continue
            
        # Skip if max depth reached
        if depth > max_depth:
            url_queue.mark_done(current_url)
            continue
            
        logger.info(f"Crawling: {current_url} (depth {depth}/{max_depth})")
        
        try:
            # Fetch URL
            result = await fetch_url(current_url, rate_limiter)
            
            if not result:
                logger.warning(f"Failed to fetch {current_url}")
                url_queue.mark_done(current_url)
                continue
                
            final_url, html, links = result
            
            # Mark this URL as done
            url_queue.mark_done(current_url)
            domain_tracker.add_url(current_url)
            
            # Save HTML and markdown immediately
            domain_dir = os.path.join(LOGS_DIR, base_domain)
            os.makedirs(domain_dir, exist_ok=True)
            
            save_html(base_domain, final_url, html)
            
            # Convert to markdown and save immediately
            markdown = convert_html_to_markdown(html, final_url)
            save_markdown(base_domain, final_url, markdown)
            
            # Clear variables to free memory
            markdown = None
            
            # If not at max depth, add new links to queue
            if depth < max_depth:
                # Only add valid page URLs
                valid_links = [link for link in links if is_valid_page_url(link)]
                url_queue.add_many(valid_links, depth + 1)
                
                # Log progress
                logger.info(f"Found {len(valid_links)} new links from {current_url}")
                logger.info(f"Queue size: {url_queue.size()}, Visited: {url_queue.visited_count()}")
                
                # Clear variables to free memory
                valid_links = None
            
            # Clear variables to free memory
            html = None
            links = None
            
            # Run garbage collection occasionally (every 10 URLs)
            if url_queue.visited_count() % 10 == 0:
                gc.collect()
        
        except Exception as e:
            logger.error(f"Error processing {current_url}: {str(e)}")
            url_queue.mark_done(current_url)
    
    # Save external links
    save_external_links(base_domain)
    
    # Final garbage collection
    gc.collect()
    
    return set(url_queue.visited)

async def crawl_website_with_error_handling(website: str, max_depth: int, max_urls: int, rate_limit: float) -> None:
    """Crawl a website with proper error handling."""
    try:
        logger.info(f"Processing website: {website}")
        processed_urls = await crawl_domain(website, max_depth, max_urls, rate_limit)
        logger.info(f"Completed processing {website} - crawled {len(processed_urls)} pages")
    except Exception as e:
        logger.error(f"Error processing website {website}: {e}")
        import traceback
        logger.error(traceback.format_exc())

async def process_csv_file(csv_file: str, max_depth: int, max_urls: int, rate_limit: float) -> None:
    """Process URLs from a CSV file with improved error handling."""
    global domain_tracker
    
    # Initialize domain tracker
    domain_tracker = DomainFailureTracker()
    domain_tracker.max_urls_per_domain = max_urls
    
    try:
        # Read URLs from CSV file
        websites = []
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'website' in row:
                    websites.append(row['website'])
        
        if not websites:
            logger.error(f"No websites found in CSV file: {csv_file}")
            return
            
        logger.info(f"Found {len(websites)} websites in CSV file")
        
        # Process websites concurrently with error handling
        tasks = []
        for website in websites:
            task = asyncio.create_task(
                crawl_website_with_error_handling(website, max_depth, max_urls, rate_limit)
            )
            tasks.append(task)
        
        # Add error callbacks to tasks
        for task in tasks:
            task.add_done_callback(handle_async_exceptions)
        
        # Wait for all tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True)
        
    except Exception as e:
        logger.error(f"Error processing CSV file {csv_file}: {e}")

async def main():
    """Main entry point for the crawler with improved error handling and memory optimization."""
    # Load configuration first, before any directory creation or logging setup
    global CONCURRENT_WORKERS, REQUEST_TIMEOUT, CURL_AVAILABLE, RESPECT_ROBOTS_TXT
    
    # Parse command line arguments to see if we need a different config file
    parser = argparse.ArgumentParser(description="DevCrawler - Web crawler for various websites")
    parser.add_argument("--config", type=str, help="Path to configuration file", default="config.yaml")
    
    # Just parse the config argument for now
    args, _ = parser.parse_known_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Now set up directories and logging based on the loaded config
    setup_directories_and_logging()
    
    # Set garbage collection thresholds for more aggressive memory management
    gc.set_threshold(700, 10, 5)  # More aggressive than default values
    
    # Parse all command line arguments now that we have logging configured
    parser.add_argument("--csv", type=str, help="CSV file with websites to crawl", default=config['files']['input_csv'])
    parser.add_argument("--depth", type=int, help="Maximum crawl depth", default=MAX_CRAWL_DEPTH)
    parser.add_argument("--max-urls", type=int, help="Maximum URLs per domain", default=MAX_URLS_PER_DOMAIN)
    parser.add_argument("--rate-limit", type=float, help="Requests per second per domain", default=RATE_LIMIT_PER_SECOND)
    parser.add_argument("--workers", type=int, help="Number of concurrent workers", default=CONCURRENT_WORKERS)
    parser.add_argument("--timeout", type=int, help="Request timeout in seconds", default=REQUEST_TIMEOUT)
    parser.add_argument("--respect-robots", action="store_true", help="Respect robots.txt", default=RESPECT_ROBOTS_TXT)
    parser.add_argument("--no-respect-robots", action="store_false", dest="respect_robots", help="Ignore robots.txt")
    
    args = parser.parse_args()
    
    # Update global settings (command line args override config file)
    CONCURRENT_WORKERS = args.workers
    REQUEST_TIMEOUT = args.timeout
    RESPECT_ROBOTS_TXT = args.respect_robots
    
    # Try to set console to UTF-8 mode on Windows
    if sys.platform.startswith("win"):
        try:
            if hasattr(ctypes, 'windll'):
                kernel32 = ctypes.windll.kernel32
                kernel32.SetConsoleCP(65001)  # Set console input to UTF-8
                kernel32.SetConsoleOutputCP(65001)  # Set console output to UTF-8
                logger.info("Set console to UTF-8 mode")
        except Exception:
            logger.warning("Failed to set console to UTF-8 mode")
    
    # Check if curl is available
    CURL_AVAILABLE = check_curl_available()
    if not CURL_AVAILABLE:
        logger.warning("curl command is not available. Will use Python requests as fallback.")
        # Make sure the requests library is installed
        try:
            import requests
        except ImportError:
            logger.error("Neither curl nor requests library is available. Cannot proceed.")
            return
    
    # Log starting information
    logger.info("DevCrawler starting")
    logger.info(f"Author: Bhaskar Dev (https://www.linkedin.com/in/bhaskar-dev/)")
    logger.info(f"Configuration: max_depth={args.depth}, max_urls={args.max_urls}, rate_limit={args.rate_limit}")
    
    # Set up task handling to catch exceptions
    loop = asyncio.get_event_loop()
    loop.set_exception_handler(lambda loop, context: logger.error(f"Async error: {context['message']}"))
    
    try:
        # Process CSV file
        await process_csv_file(args.csv, args.depth, args.max_urls, args.rate_limit)
    except Exception as e:
        logger.error(f"Error in main process: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # Clean up
        if aiohttp_session and not aiohttp_session.closed:
            await aiohttp_session.close()
        
        if browser:
            try:
                await browser.close()
            except Exception as e:
                logger.error(f"Error closing browser: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Crawler stopped by user")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        import traceback
        logger.error(traceback.format_exc())