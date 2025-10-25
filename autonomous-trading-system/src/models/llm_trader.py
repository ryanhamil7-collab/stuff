import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Dict, List, Optional
import json
from src.utils import log, config
from src.models.llm_schemas import TradingSignal, validate_llm_output
from src.models.thesis_templates import ThesisPromptTemplate, StructuredThesis

class LLMTrader:
    
    def __init__(self, model_name: str = None, use_structured_thesis: bool = True):
        if model_name is None:
            model_name = config.get('llm.model_name', 'mistralai/Mistral-7B-Instruct-v0.2')
        
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_structured_thesis = use_structured_thesis
        self.thesis_template = ThesisPromptTemplate()
        
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
        market_regime: str = "unknown"
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
            if self.use_structured_thesis:
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
                decision_dict = json.loads(json_str)
                
                if 'action' in decision_dict and 'confidence' in decision_dict and 'symbol' in decision_dict:
                    try:
                        validated = validate_llm_output(decision_dict, TradingSignal)
                        return {
                            'action': validated.action,
                            'confidence': validated.confidence,
                            'reasoning': validated.reasoning,
                            'risk_level': 'MEDIUM',
                            'time_horizon': 'MEDIUM'
                        }
                    except ValueError as ve:
                        log.warning(f"LLM output validation failed: {str(ve)}")
                        log.warning("Falling back to unvalidated output")
                        return decision_dict
            
            return self._extract_decision_from_text(decision_text)
            
        except Exception as e:
            log.error(f"Error parsing decision: {str(e)}")
            return {
                'action': 'HOLD',
                'confidence': 0.5,
                'reasoning': 'Unable to parse LLM response',
                'risk_level': 'MEDIUM',
                'time_horizon': 'MEDIUM'
            }
    
    def _extract_decision_from_text(self, text: str) -> Dict:
        text_lower = text.lower()
        
        if 'buy' in text_lower:
            action = 'BUY'
        elif 'sell' in text_lower:
            action = 'SELL'
        else:
            action = 'HOLD'
        
        confidence = 0.5
        if 'high confidence' in text_lower or 'strong' in text_lower:
            confidence = 0.8
        elif 'low confidence' in text_lower or 'weak' in text_lower:
            confidence = 0.3
        
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
        symbols_data: Dict[str, Dict]
    ) -> Dict[str, Dict]:
        decisions = {}
        
        for symbol, data in symbols_data.items():
            decision = self.generate_trading_decision(
                symbol,
                data.get('technical', {}),
                data.get('sentiment', {}),
                data.get('alpha', {}),
                data.get('market_regime', 'unknown')
            )
            decisions[symbol] = decision
        
        return decisions
