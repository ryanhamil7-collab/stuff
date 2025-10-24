from typing import Dict, List, Tuple
from pydantic import BaseModel, Field
from datetime import datetime
import numpy as np
from src.utils import log

class MarketTensionHypothesis(BaseModel):
    """
    Structured hypothesis for market tension with adversarial critique.
    """
    hypothesis_id: str = Field(..., description="Unique identifier")
    tension_type: str = Field(..., description="Type of market tension (e.g., mechanical flows, behavioral herding)")
    description: str = Field(..., max_length=300, description="Clear description of the hypothesis")
    
    variables: List[str] = Field(..., min_items=1, max_items=5, description="Observable variables")
    data_sources: List[str] = Field(..., min_items=1, description="Data provenance (e.g., yfinance volume)")
    expected_signs: Dict[str, str] = Field(..., description="Expected direction for each variable (+/-)")
    time_horizons: List[str] = Field(..., description="Expected horizons (e.g., 1d, 5d, 20d)")
    
    falsification_tests: List[str] = Field(..., min_items=1, max_items=3, description="Tests to falsify hypothesis")
    regime_breaks: List[str] = Field(..., description="Conditions where hypothesis breaks down")
    
    critique: str = Field(..., max_length=500, description="Adversarial critique of the hypothesis")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in hypothesis")

class HypothesisGenerator:
    """
    LLM-as-Reasoning-Amplifier for hypothesis generation.
    
    Uses LLM to generate market tension hypotheses with adversarial critiques,
    avoiding uncalibrated pitches and building durable edges.
    """
    
    def __init__(self, llm_trader=None):
        self.llm_trader = llm_trader
        self.hypothesis_history = []
        log.info("HypothesisGenerator initialized")
    
    def create_hypothesis_prompt(self, symbol: str, market_data: Dict, context: str = "") -> str:
        """
        Create prompt for hypothesis generation.
        
        Args:
            symbol: Stock symbol
            market_data: Recent market data
            context: Additional context
        
        Returns:
            Structured prompt
        """
        prompt = f"""Act as a skeptical quantitative researcher. Generate 3 MARKET TENSION HYPOTHESES for {symbol}.

CRITICAL RULES:
1. Each hypothesis must be FALSIFIABLE with specific tests
2. Identify variables with clear data sources (e.g., yfinance volume, price)
3. Specify expected signs (+/-) and time horizons
4. Provide ADVERSARIAL CRITIQUE for each hypothesis
5. Identify regime breaks where hypothesis fails

MARKET DATA:
- Current Price: ${market_data.get('Close', 0):.2f}
- Volume: {market_data.get('Volume', 0):,.0f}
- Volatility (20d): {market_data.get('Volatility_20d', 0)*100:.2f}%
- RSI: {market_data.get('RSI', 50):.2f}
- MACD: {market_data.get('MACD', 0):.4f}

{context}

Generate 3 hypotheses following this format:

=== HYPOTHESIS 1: [Tension Type] ===
Description: [Clear description of market tension, e.g., "Institutional rebalancing creates predictable flows"]

Variables:
- [Variable 1]: [Data source] → Expected sign: [+/-]
- [Variable 2]: [Data source] → Expected sign: [+/-]

Time Horizons: [1d, 5d, 20d]

Falsification Tests:
1. [Specific test, e.g., "If volume < 1M for 3 days, hypothesis invalid"]
2. [Another test]

Regime Breaks:
- [Condition where hypothesis fails, e.g., "During earnings announcements"]
- [Another condition]

Adversarial Critique:
[Challenge the hypothesis: What could go wrong? What assumptions are weak? 
What alternative explanations exist? Be harsh and specific.]

Confidence: [0.0-1.0]

=== HYPOTHESIS 2: [Different Tension Type] ===
[Repeat format]

=== HYPOTHESIS 3: [Different Tension Type] ===
[Repeat format]

TENSION TYPES TO CONSIDER:
- Mechanical flows (rebalancing, index inclusion/exclusion)
- Behavioral herding (momentum chasing, panic selling)
- Information asymmetry (insider trading, analyst upgrades)
- Liquidity dynamics (bid-ask spreads, market depth)
- Cross-asset correlations (sector rotation, risk-on/risk-off)

Output in structured format."""
        
        return prompt
    
    def parse_hypotheses(self, response: str) -> List[MarketTensionHypothesis]:
        """
        Parse LLM response into structured hypotheses.
        
        Args:
            response: LLM response text
        
        Returns:
            List of MarketTensionHypothesis objects
        """
        import re
        
        hypotheses = []
        
        hypothesis_blocks = re.split(r'=== HYPOTHESIS \d+:', response)
        
        for i, block in enumerate(hypothesis_blocks[1:], 1):
            try:
                tension_type_match = re.search(r'\[(.*?)\]', block)
                tension_type = tension_type_match.group(1) if tension_type_match else "Unknown"
                
                description_match = re.search(r'Description:\s*(.+?)(?=Variables:|$)', block, re.DOTALL)
                description = description_match.group(1).strip() if description_match else ""
                
                variables = []
                variables_section = re.search(r'Variables:(.*?)(?=Time Horizons:|$)', block, re.DOTALL)
                if variables_section:
                    var_lines = variables_section.group(1).strip().split('\n')
                    for line in var_lines:
                        if '-' in line:
                            var_name = line.split(':')[0].strip(' -')
                            variables.append(var_name)
                
                data_sources = ['yfinance', 'market_data']
                
                expected_signs = {}
                for var in variables:
                    sign_match = re.search(rf'{var}.*?Expected sign:\s*([+\-])', block)
                    if sign_match:
                        expected_signs[var] = sign_match.group(1)
                
                horizons_match = re.search(r'Time Horizons:\s*\[(.*?)\]', block)
                time_horizons = horizons_match.group(1).split(',') if horizons_match else ['1d', '5d', '20d']
                time_horizons = [h.strip() for h in time_horizons]
                
                falsification_tests = []
                falsification_section = re.search(r'Falsification Tests:(.*?)(?=Regime Breaks:|$)', block, re.DOTALL)
                if falsification_section:
                    test_lines = falsification_section.group(1).strip().split('\n')
                    for line in test_lines:
                        if line.strip() and (line.strip()[0].isdigit() or line.strip().startswith('-')):
                            test = re.sub(r'^\d+\.\s*|\-\s*', '', line.strip())
                            if test:
                                falsification_tests.append(test)
                
                regime_breaks = []
                regime_section = re.search(r'Regime Breaks:(.*?)(?=Adversarial Critique:|$)', block, re.DOTALL)
                if regime_section:
                    break_lines = regime_section.group(1).strip().split('\n')
                    for line in break_lines:
                        if line.strip() and line.strip().startswith('-'):
                            regime_breaks.append(line.strip(' -'))
                
                critique_match = re.search(r'Adversarial Critique:\s*(.+?)(?=Confidence:|$)', block, re.DOTALL)
                critique = critique_match.group(1).strip() if critique_match else "No critique provided"
                
                confidence_match = re.search(r'Confidence:\s*(0?\.\d+|1\.0)', block)
                confidence = float(confidence_match.group(1)) if confidence_match else 0.5
                
                hypothesis = MarketTensionHypothesis(
                    hypothesis_id=f"H{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    tension_type=tension_type,
                    description=description[:300],
                    variables=variables[:5],
                    data_sources=data_sources,
                    expected_signs=expected_signs,
                    time_horizons=time_horizons,
                    falsification_tests=falsification_tests[:3],
                    regime_breaks=regime_breaks,
                    critique=critique[:500],
                    confidence=confidence
                )
                
                hypotheses.append(hypothesis)
                
            except Exception as e:
                log.error(f"Error parsing hypothesis {i}: {str(e)}")
                continue
        
        return hypotheses
    
    def generate_hypotheses(self, symbol: str, market_data: Dict, context: str = "") -> List[MarketTensionHypothesis]:
        """
        Generate market tension hypotheses using LLM.
        
        Args:
            symbol: Stock symbol
            market_data: Recent market data
            context: Additional context
        
        Returns:
            List of structured hypotheses
        """
        log.info(f"Generating hypotheses for {symbol}")
        
        prompt = self.create_hypothesis_prompt(symbol, market_data, context)
        
        if self.llm_trader and self.llm_trader.model is not None:
            try:
                inputs = self.llm_trader.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
                inputs = {k: v.to(self.llm_trader.device) for k, v in inputs.items()}
                
                import torch
                with torch.no_grad():
                    outputs = self.llm_trader.model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        temperature=0.8,
                        do_sample=True,
                        top_p=0.9
                    )
                
                response = self.llm_trader.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                hypotheses = self.parse_hypotheses(response)
                
                self.hypothesis_history.extend(hypotheses)
                
                log.info(f"Generated {len(hypotheses)} hypotheses for {symbol}")
                
                return hypotheses
                
            except Exception as e:
                log.error(f"Error generating hypotheses: {str(e)}")
                return self._generate_fallback_hypotheses(symbol, market_data)
        else:
            log.warning("LLM not available, using fallback hypotheses")
            return self._generate_fallback_hypotheses(symbol, market_data)
    
    def _generate_fallback_hypotheses(self, symbol: str, market_data: Dict) -> List[MarketTensionHypothesis]:
        """
        Generate simple fallback hypotheses when LLM is unavailable.
        """
        rsi = market_data.get('RSI', 50)
        volume_ratio = market_data.get('Volume_Ratio', 1.0)
        
        hypotheses = []
        
        if rsi < 30:
            h1 = MarketTensionHypothesis(
                hypothesis_id=f"H1_fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tension_type="Mean Reversion",
                description="Oversold conditions suggest mean reversion opportunity",
                variables=["RSI", "Price"],
                data_sources=["yfinance"],
                expected_signs={"RSI": "+", "Price": "+"},
                time_horizons=["1d", "5d"],
                falsification_tests=["If RSI stays below 30 for >5 days, hypothesis invalid"],
                regime_breaks=["During strong downtrends", "During earnings"],
                critique="Mean reversion may not work in trending markets. RSI alone is insufficient.",
                confidence=0.6
            )
            hypotheses.append(h1)
        
        if volume_ratio > 2.0:
            h2 = MarketTensionHypothesis(
                hypothesis_id=f"H2_fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tension_type="Volume Breakout",
                description="High volume suggests institutional interest and potential breakout",
                variables=["Volume", "Price"],
                data_sources=["yfinance"],
                expected_signs={"Volume": "+", "Price": "+"},
                time_horizons=["1d", "5d"],
                falsification_tests=["If price doesn't move >2% within 2 days, hypothesis invalid"],
                regime_breaks=["During low volatility periods"],
                critique="High volume can also indicate distribution. Need price confirmation.",
                confidence=0.5
            )
            hypotheses.append(h2)
        
        return hypotheses
    
    def evaluate_hypothesis(self, hypothesis: MarketTensionHypothesis, current_data: Dict) -> Dict:
        """
        Evaluate hypothesis against current data.
        
        Args:
            hypothesis: Hypothesis to evaluate
            current_data: Current market data
        
        Returns:
            Evaluation results
        """
        results = {
            'hypothesis_id': hypothesis.hypothesis_id,
            'passed_tests': 0,
            'failed_tests': 0,
            'regime_break': False,
            'score': 0.0
        }
        
        for test in hypothesis.falsification_tests:
            try:
                if 'volume' in test.lower():
                    volume = current_data.get('Volume', 0)
                    if 'volume < 1M' in test.lower() and volume < 1000000:
                        results['failed_tests'] += 1
                    else:
                        results['passed_tests'] += 1
                else:
                    results['passed_tests'] += 1
            except:
                pass
        
        total_tests = len(hypothesis.falsification_tests)
        if total_tests > 0:
            results['score'] = results['passed_tests'] / total_tests
        
        return results
