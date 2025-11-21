#!/usr/bin/env python3

import os
import warnings
from typing import Optional, Dict, Any
from transformers import MarianMTModel, MarianTokenizer
import torch

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

class TranslationModule:
    """
    Translation module that compares source and target languages
    and only translates when they are different.
    """
    
    # Language code mapping for Whisper and MarianMT
    LANGUAGE_CODES = {
        'yue': 'zh',  # Cantonese -> Chinese (MarianMT uses 'zh')
        'zh': 'zh',   # Chinese
        'en': 'en',   # English
        'ja': 'ja',   # Japanese
        'ko': 'ko',   # Korean
        'es': 'es',   # Spanish
        'fr': 'fr',   # French
        'de': 'de',   # German
    }
    
    def __init__(self, device: str = None):
        """
        Initialize the translation module.
        
        Args:
            device: Device to use ('cuda' or 'cpu'). Auto-detect if None.
        """
        print("[TRANSLATION] Initializing Translation Module...")
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"[TRANSLATION] Using device: {self.device}")
        
        # Cache for loaded models (model_name -> (model, tokenizer))
        self.model_cache = {}
        
        # Supported translation pairs (source -> target)
        self.supported_pairs = {
            ('zh', 'en'): 'Helsinki-NLP/opus-mt-zh-en',
            ('en', 'zh'): 'Helsinki-NLP/opus-mt-en-zh',
            ('zh', 'ja'): 'Helsinki-NLP/opus-mt-zh-ja',
            ('ja', 'zh'): 'Helsinki-NLP/opus-mt-ja-zh',
            ('zh', 'ko'): 'Helsinki-NLP/opus-mt-zh-ko',
            ('ko', 'zh'): 'Helsinki-NLP/opus-mt-ko-zh',
            ('en', 'ja'): 'Helsinki-NLP/opus-mt-en-jap',
            ('ja', 'en'): 'Helsinki-NLP/opus-mt-jap-en',
            ('en', 'ko'): 'Helsinki-NLP/opus-mt-en-ko',
            ('ko', 'en'): 'Helsinki-NLP/opus-mt-ko-en',
            ('en', 'es'): 'Helsinki-NLP/opus-mt-en-es',
            ('es', 'en'): 'Helsinki-NLP/opus-mt-es-en',
            ('en', 'fr'): 'Helsinki-NLP/opus-mt-en-fr',
            ('fr', 'en'): 'Helsinki-NLP/opus-mt-fr-en',
            ('en', 'de'): 'Helsinki-NLP/opus-mt-en-de',
            ('de', 'en'): 'Helsinki-NLP/opus-mt-de-en',
        }
        
        print("[TRANSLATION] Translation Module initialized successfully")
        print(f"[TRANSLATION] Supported language pairs: {len(self.supported_pairs)}")
    
    def normalize_language_code(self, lang_code: str) -> str:
        """
        Normalize language code to MarianMT format.
        
        Args:
            lang_code: Language code (e.g., 'yue', 'zh', 'en')
            
        Returns:
            Normalized language code
        """
        return self.LANGUAGE_CODES.get(lang_code.lower(), lang_code.lower())
    
    def should_translate(self, source_lang: str, target_lang: str) -> bool:
        """
        Check if translation is needed by comparing source and target languages.
        
        Args:
            source_lang: Source language code (e.g., 'yue', 'zh', 'en')
            target_lang: Target language code
            
        Returns:
            True if translation is needed, False otherwise
        """
        # Normalize language codes
        source_normalized = self.normalize_language_code(source_lang)
        target_normalized = self.normalize_language_code(target_lang)
        
        # Check if languages are the same
        if source_normalized == target_normalized:
            print(f"[TRANSLATION] Source and target languages are the same ({source_normalized}), no translation needed")
            return False
        
        print(f"[TRANSLATION] Languages differ: {source_normalized} -> {target_normalized}, translation needed")
        return True
    
    def get_model_name(self, source_lang: str, target_lang: str) -> Optional[str]:
        """
        Get the appropriate model name for a language pair.
        
        Args:
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Model name or None if not supported
        """
        source_normalized = self.normalize_language_code(source_lang)
        target_normalized = self.normalize_language_code(target_lang)
        
        pair = (source_normalized, target_normalized)
        model_name = self.supported_pairs.get(pair)
        
        if model_name:
            print(f"[TRANSLATION] Found model for {source_normalized}->{target_normalized}: {model_name}")
        else:
            print(f"[TRANSLATION] No model found for {source_normalized}->{target_normalized}")
            print(f"[TRANSLATION] Supported pairs: {list(self.supported_pairs.keys())}")
        
        return model_name
    
    def load_model(self, model_name: str) -> tuple:
        """
        Load a translation model and tokenizer (with caching).
        
        Args:
            model_name: HuggingFace model name
            
        Returns:
            Tuple of (model, tokenizer)
        """
        # Check cache first
        if model_name in self.model_cache:
            print(f"[TRANSLATION] Using cached model: {model_name}")
            return self.model_cache[model_name]
        
        print(f"[TRANSLATION] Loading model: {model_name}")
        try:
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            model.to(self.device)
            model.eval()  # Set to evaluation mode
            
            # Cache the model
            self.model_cache[model_name] = (model, tokenizer)
            print(f"[TRANSLATION] Model loaded successfully and cached")
            
            return model, tokenizer
            
        except Exception as e:
            print(f"[TRANSLATION] Error loading model {model_name}: {e}")
            return None, None
    
    def translate_text(self, text: str, source_lang: str, target_lang: str) -> Dict[str, Any]:
        """
        Translate text from source language to target language.
        Only translates if languages are different.
        
        Args:
            text: Text to translate
            source_lang: Source language code (e.g., 'yue', 'zh', 'en')
            target_lang: Target language code
            
        Returns:
            Dictionary containing:
                - 'text': Translated text (or original if no translation needed)
                - 'translated': Boolean indicating if translation was performed
                - 'source_lang': Normalized source language
                - 'target_lang': Normalized target language
                - 'error': Error message if translation failed
        """
        result = {
            'text': text,
            'translated': False,
            'source_lang': self.normalize_language_code(source_lang),
            'target_lang': self.normalize_language_code(target_lang),
            'error': None
        }
        
        # Check if text is empty
        if not text or not text.strip():
            print("[TRANSLATION] Empty text, skipping translation")
            return result
        
        # Check if translation is needed
        if not self.should_translate(source_lang, target_lang):
            print(f"[TRANSLATION] No translation needed, returning original text")
            return result
        
        # Get appropriate model
        model_name = self.get_model_name(source_lang, target_lang)
        if not model_name:
            error_msg = f"Translation not supported for {source_lang}->{target_lang}"
            print(f"[TRANSLATION] {error_msg}")
            result['error'] = error_msg
            return result
        
        # Load model
        model, tokenizer = self.load_model(model_name)
        if model is None or tokenizer is None:
            error_msg = "Failed to load translation model"
            print(f"[TRANSLATION] {error_msg}")
            result['error'] = error_msg
            return result
        
        # Perform translation
        try:
            print(f"[TRANSLATION] Translating: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Translate
            with torch.no_grad():
                translated = model.generate(**inputs)
            
            # Decode
            translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
            
            print(f"[TRANSLATION] Translation complete: '{translated_text[:50]}{'...' if len(translated_text) > 50 else ''}'")
            
            result['text'] = translated_text
            result['translated'] = True
            
            return result
            
        except Exception as e:
            error_msg = f"Translation error: {str(e)}"
            print(f"[TRANSLATION] {error_msg}")
            result['error'] = error_msg
            return result
    
    def clear_cache(self):
        """Clear the model cache to free memory."""
        print("[TRANSLATION] Clearing model cache...")
        self.model_cache.clear()
        
        # Clean up GPU memory if using CUDA
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        print("[TRANSLATION] Cache cleared")
    
    def get_supported_languages(self) -> list:
        """Get list of supported languages."""
        languages = set()
        for source, target in self.supported_pairs.keys():
            languages.add(source)
            languages.add(target)
        return sorted(list(languages))

