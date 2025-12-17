#!/usr/bin/env python3
"""
ERN URL Discovery Tool
======================
Discovers the actual URL structure of an ERN website before full scraping.

This tool helps you:
1. Fetch the main page and extract all links
2. Parse the sitemap if available
3. Identify the actual navigation structure
4. Suggest paths for the config file

Usage:
    python ern_url_discover.py https://www.erknet.org
    python ern_url_discover.py https://eurobloodnet.eu --depth 2
    python ern_url_discover.py --all  # Discover all enabled networks

Requirements:
    pip install requests beautifulsoup4
"""

import argparse
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import defaultdict, Counter
import json
import sys
import re

def get_domain(url):
    """Extract domain from URL."""
    return urlparse(url).netloc

def fetch_page(url, timeout=15):
    """Fetch a page with error handling."""
    headers = {
        'User-Agent': 'Mozilla/5.0 ERN-Discovery-Bot/1.0'
    }
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response
    except Exception as e:
        print(f"  ❌ Error fetching {url}: {e}")
        return None

def parse_sitemap(url):
    """Parse sitemap.xml and return URLs."""
    urls = []
    sitemap_urls = [
        urljoin(url, '/sitemap.xml'),
        urljoin(url, '/wp-sitemap.xml'),
        urljoin(url, '/sitemap_index.xml'),
    ]
    
    for sitemap_url in sitemap_urls:
        try:
            response = requests.get(sitemap_url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'xml')
                for loc in soup.find_all('loc'):
                    urls.append(loc.get_text(strip=True))
                if urls:
                    print(f"  📍 Found sitemap at {sitemap_url} ({len(urls)} URLs)")
                    return urls
        except Exception:
            continue
    
    return urls

def extract_nav_links(soup, base_url):
    """Extract navigation links from page."""
    nav_links = []
    
    # Look for nav elements
    for nav in soup.find_all(['nav', 'header']):
        for a in nav.find_all('a', href=True):
            href = a.get('href', '')
            text = a.get_text(strip=True)
            if href and not href.startswith(('#', 'mailto:', 'tel:', 'javascript:')):
                full_url = urljoin(base_url, href)
                if get_domain(full_url) == get_domain(base_url):
                    nav_links.append({
                        'url': full_url,
                        'text': text,
                        'path': urlparse(full_url).path
                    })
    
    return nav_links

def extract_all_links(soup, base_url):
    """Extract all internal links from page."""
    links = []
    base_domain = get_domain(base_url)
    
    for a in soup.find_all('a', href=True):
        href = a.get('href', '')
        text = a.get_text(strip=True)
        
        if href.startswith(('#', 'mailto:', 'tel:', 'javascript:')):
            continue
        
        full_url = urljoin(base_url, href)
        
        if get_domain(full_url) == base_domain:
            path = urlparse(full_url).path
            links.append({
                'url': full_url,
                'text': text[:50] if text else '',
                'path': path
            })
    
    return links

def analyze_path_structure(links):
    """Analyze the path structure to find common patterns."""
    paths = [l['path'] for l in links if l['path']]
    
    # Count first-level paths
    first_level = Counter()
    for path in paths:
        parts = [p for p in path.strip('/').split('/') if p]
        if parts:
            first_level[f"/{parts[0]}/"] += 1
    
    # Count second-level paths
    second_level = Counter()
    for path in paths:
        parts = [p for p in path.strip('/').split('/') if p]
        if len(parts) >= 2:
            second_level[f"/{parts[0]}/{parts[1]}/"] += 1
    
    return {
        'first_level': first_level.most_common(20),
        'second_level': second_level.most_common(20)
    }

def discover_website(url, depth=1):
    """Discover website structure."""
    print(f"\n{'='*60}")
    print(f"🔍 Discovering: {url}")
    print(f"{'='*60}")
    
    base_domain = get_domain(url)
    results = {
        'url': url,
        'domain': base_domain,
        'sitemap_urls': [],
        'nav_links': [],
        'all_links': [],
        'suggested_paths': [],
        'path_analysis': {}
    }
    
    # Try sitemap first
    print("\n📍 Checking sitemap...")
    sitemap_urls = parse_sitemap(url)
    results['sitemap_urls'] = sitemap_urls
    
    # Fetch main page
    print("\n📄 Fetching main page...")
    response = fetch_page(url)
    
    if not response:
        return results
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extract navigation
    print("\n🧭 Analyzing navigation...")
    nav_links = extract_nav_links(soup, url)
    results['nav_links'] = nav_links
    
    if nav_links:
        print(f"  Found {len(nav_links)} navigation links:")
        for link in nav_links[:15]:
            print(f"    • {link['path']}: {link['text'][:40]}")
    
    # Extract all links
    print("\n🔗 Extracting all links...")
    all_links = extract_all_links(soup, url)
    results['all_links'] = all_links
    print(f"  Found {len(all_links)} internal links")
    
    # Combine with sitemap URLs
    all_paths = set()
    for link in all_links:
        all_paths.add(link['path'])
    for sitemap_url in sitemap_urls:
        all_paths.add(urlparse(sitemap_url).path)
    
    # Analyze structure
    print("\n📊 Analyzing path structure...")
    analysis = analyze_path_structure(all_links)
    results['path_analysis'] = analysis
    
    print("\n  Top first-level paths:")
    for path, count in analysis['first_level'][:10]:
        print(f"    {path}: {count} links")
    
    # Generate suggested paths for config
    suggested = []
    
    # Add main paths from analysis
    for path, count in analysis['first_level']:
        if count >= 2:
            # Skip obvious non-content paths
            skip_patterns = ['wp-', 'admin', 'login', 'cart', 'search', 'tag', 'author']
            if not any(p in path.lower() for p in skip_patterns):
                suggested.append(path)
    
    # Add navigation paths
    for link in nav_links:
        if link['path'] and link['path'] not in suggested:
            suggested.append(link['path'])
    
    results['suggested_paths'] = suggested[:30]
    
    # Print suggestions
    print("\n✅ Suggested paths for config:")
    print("```json")
    print('"guidelines_paths": [')
    for path in results['suggested_paths'][:20]:
        print(f'    "{path}",')
    print(']')
    print("```")
    
    # Fetch deeper if requested
    if depth > 1:
        print(f"\n🔍 Exploring depth {depth}...")
        explored = {url}
        to_explore = [l['url'] for l in all_links[:20]]
        
        for explore_url in to_explore:
            if explore_url in explored:
                continue
            explored.add(explore_url)
            
            print(f"  📄 {urlparse(explore_url).path}")
            resp = fetch_page(explore_url)
            if resp:
                sub_soup = BeautifulSoup(resp.text, 'html.parser')
                sub_links = extract_all_links(sub_soup, explore_url)
                for link in sub_links:
                    if link['path'] not in all_paths:
                        all_paths.add(link['path'])
                        results['all_links'].append(link)
    
    return results

def load_config(config_path='ern_config.json'):
    """Load ERN config file."""
    try:
        with open(config_path) as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return None

def discover_all_networks(config_path='ern_config.json'):
    """Discover all enabled networks."""
    config = load_config(config_path)
    if not config:
        return
    
    networks = config.get('networks', {})
    results = {}
    
    for network_id, info in networks.items():
        if not info.get('scrape', False):
            continue
        
        website = info.get('website')
        if not website:
            continue
        
        results[network_id] = discover_website(website)
        print("\n")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='ERN URL Discovery Tool')
    parser.add_argument('url', nargs='?', help='Website URL to discover')
    parser.add_argument('--depth', type=int, default=1, help='Discovery depth (1-3)')
    parser.add_argument('--all', action='store_true', help='Discover all enabled networks')
    parser.add_argument('-c', '--config', default='ern_config.json', help='Config file path')
    parser.add_argument('-o', '--output', help='Output JSON file')
    
    args = parser.parse_args()
    
    if args.all:
        results = discover_all_networks(args.config)
    elif args.url:
        results = discover_website(args.url, depth=args.depth)
    else:
        parser.print_help()
        return
    
    if args.output and results:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Results saved to {args.output}")

if __name__ == '__main__':
    main()