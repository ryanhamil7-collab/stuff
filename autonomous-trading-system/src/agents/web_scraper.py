"""
Web Scraper for Symbol Discovery and Market Intelligence

Scrapes multiple sources to find trending stocks, news, and market sentiment:
- Yahoo Finance trending stocks
- Finviz screener for momentum stocks
- Reddit WallStreetBets for retail sentiment
- Google Trends for search volume
- News aggregation for market events
"""

import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import time
from src.utils import log, config

class WebScraper:
    """
    Headless web scraper for discovering trading opportunities
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.rate_limit_delay = 2  # seconds between requests
        log.info("WebScraper initialized")
    
    def scrape_yahoo_trending(self) -> List[Dict]:
        """
        Scrape Yahoo Finance trending stocks
        
        Returns:
            List of trending stocks with metadata
        """
        log.info("Scraping Yahoo Finance trending stocks...")
        
        try:
            url = "https://finance.yahoo.com/trending-tickers"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            trending_stocks = []
            
            table = soup.find('table')
            if table:
                rows = table.find_all('tr')[1:]  # Skip header
                
                for row in rows[:20]:  # Top 20
                    cols = row.find_all('td')
                    if len(cols) >= 3:
                        symbol = cols[0].text.strip()
                        name = cols[1].text.strip() if len(cols) > 1 else ""
                        price_change = cols[2].text.strip() if len(cols) > 2 else "0%"
                        
                        trending_stocks.append({
                            'symbol': symbol,
                            'name': name,
                            'price_change': price_change,
                            'source': 'yahoo_trending',
                            'timestamp': datetime.now()
                        })
            
            log.info(f"✓ Found {len(trending_stocks)} trending stocks from Yahoo")
            return trending_stocks
            
        except Exception as e:
            log.error(f"Error scraping Yahoo trending: {str(e)}")
            return []
    
    def scrape_finviz_screener(self, criteria: str = "ta_topgainers") -> List[Dict]:
        """
        Scrape Finviz stock screener
        
        Args:
            criteria: Screener criteria (ta_topgainers, ta_toplosers, ta_mostvolatile, etc.)
        
        Returns:
            List of stocks matching criteria
        """
        log.info(f"Scraping Finviz screener ({criteria})...")
        
        try:
            url = f"https://finviz.com/screener.ashx?v=111&f={criteria}"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            stocks = []
            
            table = soup.find('table', {'class': 'table-light'})
            if table:
                rows = table.find_all('tr')[1:]  # Skip header
                
                for row in rows[:30]:  # Top 30
                    cols = row.find_all('td')
                    if len(cols) >= 2:
                        ticker_link = cols[1].find('a')
                        if ticker_link:
                            symbol = ticker_link.text.strip()
                            
                            stocks.append({
                                'symbol': symbol,
                                'source': f'finviz_{criteria}',
                                'timestamp': datetime.now()
                            })
            
            log.info(f"✓ Found {len(stocks)} stocks from Finviz {criteria}")
            time.sleep(self.rate_limit_delay)
            return stocks
            
        except Exception as e:
            log.error(f"Error scraping Finviz: {str(e)}")
            return []
    
    def scrape_reddit_wsb(self, limit: int = 50) -> List[Dict]:
        """
        Scrape Reddit WallStreetBets for mentioned tickers
        
        Args:
            limit: Number of posts to analyze
        
        Returns:
            List of mentioned tickers with sentiment
        """
        log.info("Scraping Reddit WallStreetBets...")
        
        try:
            url = f"https://www.reddit.com/r/wallstreetbets/hot.json?limit={limit}"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            ticker_mentions = {}
            
            for post in data.get('data', {}).get('children', []):
                post_data = post.get('data', {})
                title = post_data.get('title', '')
                selftext = post_data.get('selftext', '')
                score = post_data.get('score', 0)
                
                import re
                text = f"{title} {selftext}"
                tickers = re.findall(r'\$([A-Z]{1,5})\b|\b([A-Z]{2,5})\b', text)
                
                for ticker_match in tickers:
                    ticker = ticker_match[0] or ticker_match[1]
                    if ticker and len(ticker) <= 5:  # Valid ticker length
                        if ticker not in ticker_mentions:
                            ticker_mentions[ticker] = {
                                'symbol': ticker,
                                'mentions': 0,
                                'total_score': 0,
                                'source': 'reddit_wsb'
                            }
                        ticker_mentions[ticker]['mentions'] += 1
                        ticker_mentions[ticker]['total_score'] += score
            
            sorted_tickers = sorted(
                ticker_mentions.values(),
                key=lambda x: (x['mentions'], x['total_score']),
                reverse=True
            )[:20]  # Top 20
            
            for ticker in sorted_tickers:
                ticker['timestamp'] = datetime.now()
            
            log.info(f"✓ Found {len(sorted_tickers)} mentioned tickers from Reddit")
            time.sleep(self.rate_limit_delay)
            return sorted_tickers
            
        except Exception as e:
            log.error(f"Error scraping Reddit: {str(e)}")
            return []
    
    def scrape_market_news(self) -> List[Dict]:
        """
        Scrape market news headlines from multiple sources
        
        Returns:
            List of news articles with sentiment
        """
        log.info("Scraping market news...")
        
        news_articles = []
        
        try:
            url = "https://www.marketwatch.com/latest-news"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            articles = soup.find_all('div', {'class': 'article__content'})[:10]
            
            for article in articles:
                headline_tag = article.find('h3')
                if headline_tag:
                    headline = headline_tag.text.strip()
                    link_tag = headline_tag.find('a')
                    link = link_tag.get('href', '') if link_tag else ''
                    
                    news_articles.append({
                        'headline': headline,
                        'link': link,
                        'source': 'marketwatch',
                        'timestamp': datetime.now()
                    })
            
            log.info(f"✓ Found {len(articles)} articles from MarketWatch")
            time.sleep(self.rate_limit_delay)
            
        except Exception as e:
            log.error(f"Error scraping MarketWatch: {str(e)}")
        
        return news_articles
    
    def discover_symbols(self) -> List[str]:
        """
        Discover trading symbols from all sources
        
        Returns:
            Deduplicated list of symbols ranked by relevance
        """
        log.info("=" * 80)
        log.info("DISCOVERING TRADING SYMBOLS")
        log.info("=" * 80)
        
        all_symbols = {}
        
        try:
            yahoo_stocks = self.scrape_yahoo_trending()
            for stock in yahoo_stocks:
                symbol = stock['symbol']
                if symbol not in all_symbols:
                    all_symbols[symbol] = {'score': 0, 'sources': []}
                all_symbols[symbol]['score'] += 3
                all_symbols[symbol]['sources'].append('yahoo_trending')
        except Exception as e:
            log.warning(f"Yahoo trending failed: {e}")
        
        try:
            finviz_gainers = self.scrape_finviz_screener('ta_topgainers')
            for stock in finviz_gainers:
                symbol = stock['symbol']
                if symbol not in all_symbols:
                    all_symbols[symbol] = {'score': 0, 'sources': []}
                all_symbols[symbol]['score'] += 2
                all_symbols[symbol]['sources'].append('finviz_gainers')
        except Exception as e:
            log.warning(f"Finviz gainers failed: {e}")
        
        try:
            finviz_volatile = self.scrape_finviz_screener('ta_mostvolatile')
            for stock in finviz_volatile:
                symbol = stock['symbol']
                if symbol not in all_symbols:
                    all_symbols[symbol] = {'score': 0, 'sources': []}
                all_symbols[symbol]['score'] += 2
                all_symbols[symbol]['sources'].append('finviz_volatile')
        except Exception as e:
            log.warning(f"Finviz volatile failed: {e}")
        
        try:
            reddit_tickers = self.scrape_reddit_wsb()
            for ticker in reddit_tickers:
                symbol = ticker['symbol']
                if symbol not in all_symbols:
                    all_symbols[symbol] = {'score': 0, 'sources': []}
                all_symbols[symbol]['score'] += min(ticker['mentions'], 5)
                all_symbols[symbol]['sources'].append('reddit_wsb')
        except Exception as e:
            log.warning(f"Reddit WSB failed: {e}")
        
        if not all_symbols:
            log.error("All web scraping sources failed, returning empty list")
            return []
        
        sorted_symbols = sorted(
            all_symbols.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )
        
        top_symbols = [symbol for symbol, data in sorted_symbols[:30]]
        
        log.info("=" * 80)
        log.info(f"DISCOVERED {len(top_symbols)} HIGH-POTENTIAL SYMBOLS")
        log.info("=" * 80)
        for i, symbol in enumerate(top_symbols[:10], 1):
            score = all_symbols[symbol]['score']
            sources = ', '.join(all_symbols[symbol]['sources'])
            log.info(f"{i}. {symbol} (score: {score}, sources: {sources})")
        log.info("=" * 80)
        
        return top_symbols

if __name__ == "__main__":
    scraper = WebScraper()
    symbols = scraper.discover_symbols()
    print(f"\nDiscovered {len(symbols)} symbols: {symbols}")
