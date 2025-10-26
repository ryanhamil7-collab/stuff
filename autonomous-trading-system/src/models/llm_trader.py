import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Dict, List, Optional
import json
import pandas as pd
from src.utils import log, config
from src.models.llm_schemas import TradingSignal, validate_llm_output
from src.models.thesis_templates import ThesisPromptTemplate, StructuredThesis
from src.agents.prompt_agent import PromptAgent

class LLMTrader:
    
    def __init__(self, model_name: str = None, use_structured_thesis: bool = True, use_prompt_agent: bool = True):
        if model_name is None:
            model_name = config.get('llm.model_name', 'mistralai/Mistral-7B-Instruct-v0.2')
        
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_structured_thesis = use_structured_thesis
        self.use_prompt_agent = use_prompt_agent
        self.thesis_template = ThesisPromptTemplate()
        self.prompt_agent = PromptAgent() if use_prompt_agent else None
        
        log.info(f"Loading LLM model: {model_name}")
        log.info(f"Structured thesis format: {'enabled' if use_structured_thesis else 'disabled'}")
        
        try:
            quantization = config.get('llm.quantization', '4bit')
            
            if quantization == '4bit' and torch.cuda.is_available():
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto" if torch.cuda.is_available() else None,
                    trust_remote_code=True
                )
                if not torch.cuda.is_available():
                    self.model = self.model.to(self.device)
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            log.info("LLM model loaded successfully")
            log.info(f"Model device: {self.device}")
            log.info(f"Model dtype: {self.model.dtype if hasattr(self.model, 'dtype') else 'unknown'}")
            log.info(f"Quantization: {quantization}")
            
        except Exception as e:
            log.error(f"Error loading LLM model: {str(e)}")
            self.model = None
            self.tokenizer = None
    
    def create_trading_prompt(
        self, 
        symbol: str, 
        technical_data: Dict, 
        sentiment_data: Dict,
        alpha_signals: Dict,
        market_regime: str
    ) -> str:
        prompt = f"""You are an expert quantitative trader analyzing {symbol}.

Market Context:
- Market Regime: {market_regime}
- Current Price: ${technical_data.get('current_price', 0):.2f}

Technical Indicators:
- RSI: {technical_data.get('RSI', 0):.2f}
- MACD: {technical_data.get('MACD', 0):.2f}
- MACD Signal: {technical_data.get('MACD_Signal', 0):.2f}
- SMA 20: ${technical_data.get('SMA_20', 0):.2f}
- SMA 50: ${technical_data.get('SMA_50', 0):.2f}
- Bollinger Band Position: {technical_data.get('BB_Position', 0):.2f}
- ATR: ${technical_data.get('ATR', 0):.2f}
- ADX: {technical_data.get('ADX', 0):.2f}

Sentiment Analysis:
- Overall Sentiment: {sentiment_data.get('overall_sentiment', 0):.3f}
- Signal: {sentiment_data.get('signal', 'neutral')}
- Articles Analyzed: {sentiment_data.get('articles_analyzed', 0)}

Alpha Signals:
- Combined Alpha Score: {alpha_signals.get('combined_score', 0):.3f}

Based on this comprehensive analysis, provide a trading decision in JSON format.
IMPORTANT: Your reasoning MUST reference at least one real indicator (RSI, MACD, SMA, sentiment, etc).
Do NOT invent indicators or use unrealistic factors.

{{
    "symbol": "{symbol}",
    "action": "BUY" or "SELL" or "HOLD",
    "confidence": 0.0 to 1.0,
    "reasoning": "Brief explanation referencing real indicators (max 200 chars)"
}}

Trading Decision:"""
        
        return prompt
    
    def generate_trading_decision(
        self, 
        symbol: str, 
        technical_data: Dict, 
        sentiment_data: Dict = None,
        alpha_signals: Dict = None,
        market_regime: str = "unknown",
        market_data: pd.DataFrame = None,
        portfolio_state: Dict = None,
        risk_context: Dict = None
    ) -> Dict:
        if self.model is None or self.tokenizer is None:
            log.warning("LLM model not available, using fallback logic")
            log.warning("This means the AI is NOT making decisions - only technical indicators")
            log.warning("Check if model loaded correctly or if HuggingFace token is set")
            return self._fallback_decision(technical_data)
        
        if sentiment_data is None:
            sentiment_data = {'overall_sentiment': 0.0, 'signal': 'neutral', 'articles_analyzed': 0}
        
        if alpha_signals is None:
            alpha_signals = {'combined_score': 0.0}
        
        try:
            if self.use_prompt_agent and self.prompt_agent and market_data is not None:
                prompt = self.prompt_agent.build_trading_prompt(
                    symbol=symbol,
                    market_data=market_data,
                    technical_signals=technical_data,
                    sentiment_score=sentiment_data.get('overall_sentiment', 0.0),
                    alpha_signals=alpha_signals.get('top_alphas') if isinstance(alpha_signals, dict) else None,
                    portfolio_state=portfolio_state,
                    risk_context=risk_context
                )
            elif self.use_structured_thesis:
                prompt = self.thesis_template.create_structured_prompt(
                    symbol, technical_data, sentiment_data, alpha_signals, market_regime
                )
            else:
                prompt = self.create_trading_prompt(
                    symbol, technical_data, sentiment_data, alpha_signals, market_regime
                )
            
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            max_length = config.get('llm.max_length', 2048)
            temperature = config.get('llm.temperature', 0.7)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=temperature,
                    do_sample=True,
                    top_p=config.get('llm.top_p', 0.9),
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            decision_text = response.split("Trading Decision:")[-1].strip()
            
            log.debug(f"LLM raw response for {symbol}: {decision_text[:300]}")
            
            decision = self._parse_decision(decision_text)
            
            log.info(f"LLM decision for {symbol}: {decision['action']} (confidence: {decision['confidence']:.2f})")
            
            return decision
            
        except Exception as e:
            log.error(f"Error generating LLM decision: {str(e)}")
            return self._fallback_decision(technical_data)
    
    def _parse_decision(self, decision_text: str) -> Dict:
        try:
            if '{' in decision_text and '}' in decision_text:
                json_start = decision_text.index('{')
                json_end = decision_text.rindex('}') + 1
                json_str = decision_text[json_start:json_end]
                
                try:
                    decision_dict = json.loads(json_str)
                except json.JSONDecodeError as je:
                    log.warning(f"JSON parse error at line {je.lineno}, col {je.colno}: {je.msg}")
                    log.warning(f"Attempting to repair JSON...")
                    
                    json_str_fixed = json_str
                    
                    import re
                    json_str_fixed = re.sub(r'(\w+):', r'"\1":', json_str_fixed)
                    
                    json_str_fixed = json_str_fixed.replace("'", '"')
                    
                    json_str_fixed = re.sub(r',\s*}', '}', json_str_fixed)
                    json_str_fixed = re.sub(r',\s*]', ']', json_str_fixed)
                    
                    try:
                        decision_dict = json.loads(json_str_fixed)
                        log.info("Successfully repaired JSON")
                    except json.JSONDecodeError:
                        log.warning("JSON repair failed, extracting from text")
                        return self._extract_decision_from_text(decision_text)
                
                if 'action' in decision_dict and 'confidence' in decision_dict:
                    action = str(decision_dict['action']).upper()
                    if action not in ['BUY', 'SELL', 'HOLD']:
                        log.warning(f"Invalid action '{action}', defaulting to HOLD")
                        action = 'HOLD'
                    
                    try:
                        confidence = float(decision_dict['confidence'])
                        confidence = max(0.0, min(1.0, confidence))
                    except (ValueError, TypeError):
                        log.warning(f"Invalid confidence value, defaulting to 0.5")
                        confidence = 0.5
                    
                    return {
                        'action': action,
                        'confidence': confidence,
                        'reasoning': str(decision_dict.get('reasoning', 'No reasoning provided'))[:200],
                        'risk_level': 'MEDIUM',
                        'time_horizon': 'MEDIUM'
                    }
            
            return self._extract_decision_from_text(decision_text)
            
        except Exception as e:
            log.error(f"Error parsing decision: {str(e)}")
            log.debug(f"Decision text: {decision_text[:500]}")
            return {
                'action': 'HOLD',
                'confidence': 0.5,
                'reasoning': 'Unable to parse LLM response',
                'risk_level': 'MEDIUM',
                'time_horizon': 'MEDIUM'
            }
    
    def _extract_decision_from_text(self, text: str) -> Dict:
        text_lower = text.lower()
        
        import re
        action_pattern = r'"action"\s*:\s*"?(BUY|SELL|HOLD)"?'
        action_match = re.search(action_pattern, text, re.IGNORECASE)
        
        if action_match:
            action = action_match.group(1).upper()
        else:
            sell_indicators = ['recommend sell', 'should sell', 'action: sell', 'decision: sell', 'suggest sell']
            buy_indicators = ['recommend buy', 'should buy', 'action: buy', 'decision: buy', 'suggest buy']
            hold_indicators = ['recommend hold', 'should hold', 'action: hold', 'decision: hold', 'suggest hold']
            
            if any(indicator in text_lower for indicator in sell_indicators):
                action = 'SELL'
            elif any(indicator in text_lower for indicator in buy_indicators):
                action = 'BUY'
            elif any(indicator in text_lower for indicator in hold_indicators):
                action = 'HOLD'
            else:
                action = 'HOLD'
        
        confidence_pattern = r'"confidence"\s*:\s*([0-9.]+)'
        confidence_match = re.search(confidence_pattern, text)
        
        if confidence_match:
            try:
                confidence = float(confidence_match.group(1))
                confidence = max(0.0, min(1.0, confidence))
            except:
                confidence = 0.5
        else:
            confidence = 0.5
            if 'high confidence' in text_lower or 'very confident' in text_lower:
                confidence = 0.8
            elif 'low confidence' in text_lower or 'not confident' in text_lower:
                confidence = 0.3
        
        log.debug(f"Extracted from text: action={action}, confidence={confidence}")
        
        return {
            'action': action,
            'confidence': confidence,
            'reasoning': text[:200],
            'risk_level': 'MEDIUM',
            'time_horizon': 'MEDIUM'
        }
    
    def _fallback_decision(self, technical_data: Dict) -> Dict:
        signal = technical_data.get('Signal', 0)
        rsi = technical_data.get('RSI', 50)
        
        if signal > 0 and rsi < 40:
            action = 'BUY'
            confidence = 0.6
        elif signal < 0 and rsi > 60:
            action = 'SELL'
            confidence = 0.6
        else:
            action = 'HOLD'
            confidence = 0.5
        
        return {
            'action': action,
            'confidence': confidence,
            'reasoning': 'Fallback decision based on technical indicators',
            'risk_level': 'MEDIUM',
            'time_horizon': 'MEDIUM'
        }
    
    def batch_generate_decisions(
        self, 
        symbols_data: Dict[str, Dict],
        batch_size: int = 4
    ) -> Dict[str, Dict]:
        """
        Generate trading decisions for multiple symbols using batch inference.
        
        Args:
            symbols_data: Dict mapping symbols to their data
            batch_size: Number of symbols to process in parallel (default: 4)
        
        Returns:
            Dict mapping symbols to trading decisions
        """
        if not self.model or not self.tokenizer:
            log.warning("Model not loaded, using fallback decisions")
            return {symbol: self._fallback_decision(symbol) for symbol in symbols_data}
        
        decisions = {}
        symbols = list(symbols_data.keys())
        
        log.info(f"Batch processing {len(symbols)} symbols with batch_size={batch_size}")
        
        for i in range(0, len(symbols), batch_size):
            batch_symbols = symbols[i:i+batch_size]
            batch_prompts = []
            
            for symbol in batch_symbols:
                data = symbols_data[symbol]
                
                if self.use_prompt_agent and self.prompt_agent:
                    prompt = self.prompt_agent.create_prompt(
                        symbol=symbol,
                        technical_data=data.get('technical', {}),
                        sentiment_data=data.get('sentiment', {}),
                        alpha_signals=data.get('alpha', {}),
                        market_regime=data.get('market_regime', 'unknown')
                    )
                else:
                    prompt = self.create_trading_prompt(
                        symbol,
                        data.get('technical', {}),
                        data.get('sentiment', {}),
                        data.get('alpha', {}),
                        data.get('market_regime', 'unknown')
                    )
                
                batch_prompts.append(prompt)
            
            try:
                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=config.get('llm.max_new_tokens', 256),
                        temperature=config.get('llm.temperature', 0.7),
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                
                for idx, symbol in enumerate(batch_symbols):
                    response = self.tokenizer.decode(outputs[idx], skip_special_tokens=True)
                    
                    prompt_end = response.find("Trading Decision:")
                    if prompt_end != -1:
                        response = response[prompt_end + len("Trading Decision:"):]
                    
                    decision = self._parse_decision(response, symbol)
                    decisions[symbol] = decision
                    
                    log.debug(f"Batch decision for {symbol}: {decision['action']} (confidence: {decision['confidence']:.2f})")
            
            except Exception as e:
                log.error(f"Error in batch inference: {str(e)}")
                for symbol in batch_symbols:
                    decisions[symbol] = self._fallback_decision(symbol)
        
        log.info(f"Batch processing complete: {len(decisions)} decisions generated")
        return decisions
