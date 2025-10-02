#!/usr/bin/env python3

import os
import json
import sqlite3
import pickle
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import threading
from collections import defaultdict
import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class OptimizedSpeakerDatabase:
    """Optimized database with caching and batch operations."""
    
    def __init__(self, db_path: str = "speakers.db"):
        self.db_path = db_path
        self.cache = {}  # In-memory cache for fast lookups
        self.lock = threading.Lock()
        self.init_database()
        self.load_cache()
    
    def init_database(self):
        """Initialize the speaker database with optimizations."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS speakers (
                speaker_id TEXT PRIMARY KEY,
                name TEXT,
                embedding BLOB,
                first_seen TIMESTAMP,
                last_seen TIMESTAMP,
                segment_count INTEGER,
                confidence_scores BLOB
            )
        ''')
        
        # Create index for faster lookups
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_speaker_id ON speakers(speaker_id)
        ''')
        
        conn.commit()
        conn.close()
    
    def load_cache(self):
        """Load all speakers into memory cache for fast access."""
        with self.lock:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = conn.cursor()
            
            cursor.execute("SELECT speaker_id, embedding, segment_count FROM speakers")
            results = cursor.fetchall()
            
            self.cache = {}
            for speaker_id, embedding_blob, segment_count in results:
                embedding = pickle.loads(embedding_blob)
                self.cache[speaker_id] = {
                    'embedding': embedding,
                    'segment_count': segment_count
                }
            
            conn.close()
            print(f"[DATABASE] Loaded {len(self.cache)} speakers into cache")
    
    def add_speaker(self, speaker_id: str, embedding: np.ndarray, name: str = None):
        """Add or update a speaker with ultra-fast caching."""
        with self.lock:
            # Ultra-fast: Cache only, skip database operations
            if speaker_id in self.cache:
                # Update existing speaker in cache
                self.cache[speaker_id]['segment_count'] += 1
                self.cache[speaker_id]['last_seen'] = datetime.now()
                self.cache[speaker_id]['embedding'] = embedding  # Update embedding
            else:
                # Add new speaker to cache
                self.cache[speaker_id] = {
                    'embedding': embedding,
                    'name': name or speaker_id,
                    'first_seen': datetime.now(),
                    'last_seen': datetime.now(),
                    'segment_count': 1
                }
            
            # Skip database operations for maximum speed
            # Database will be updated only on session reset
    
    def find_similar_speaker(self, embedding: np.ndarray, threshold: float = 0.8) -> Optional[str]:
        """Find the most similar speaker using cached embeddings."""
        if not self.cache:
            return None
        
        best_match = None
        best_similarity = 0.0
        
        # Use cached embeddings for fast comparison
        for speaker_id, data in self.cache.items():
            stored_embedding = data['embedding']
            
            # Calculate cosine similarity
            similarity = cosine_similarity(
                embedding.reshape(1, -1), 
                stored_embedding.reshape(1, -1)
            )[0][0]
            
            print(f"[SPEAKER_RECOGNITION] Similarity with {speaker_id}: {similarity:.3f} (threshold: {threshold})")
            
            if similarity > threshold and similarity > best_similarity:
                best_similarity = similarity
                best_match = speaker_id
        
        if best_match:
            print(f"[SPEAKER_RECOGNITION] Best match: {best_match} (similarity: {best_similarity:.3f})")
        else:
            print(f"[SPEAKER_RECOGNITION] No match found (best similarity: {best_similarity:.3f})")
        
        return best_match
    
    
    
    def check_speaker_consistency(self, speaker_id: str, embedding: np.ndarray) -> bool:
        """Check if the speaker assignment is consistent with existing memory."""
        if speaker_id not in self.session_speaker_embeddings:
            return True  # New speaker, no consistency check needed
        
        # Check similarity with existing embeddings for this speaker
        similarities = []
        for stored_embedding in self.session_speaker_embeddings[speaker_id]:
            similarity = cosine_similarity([embedding], [stored_embedding])[0][0]
            similarities.append(similarity)
        
        if similarities:
            avg_similarity = np.mean(similarities)
            print(f"[OPTIMIZED] Consistency check for {speaker_id}: avg_similarity={avg_similarity:.3f}")
            
            # If similarity is too low, this might be a different speaker
            if avg_similarity < 0.3:
                print(f"[OPTIMIZED] WARNING: Low consistency for {speaker_id}, might be different speaker")
                return False
        
        return True
    
    def force_speaker_consistency(self, original_speaker: str, embedding: np.ndarray) -> str:
        """Force consistent speaker assignment to prevent flipping."""
        # First check if this diarization speaker already has a consistent mapping
        if original_speaker in self.diarization_speaker_map:
            mapped_speaker = self.diarization_speaker_map[original_speaker]
            
            # Check if the mapped speaker is consistent with this embedding
            if self.check_speaker_consistency(mapped_speaker, embedding):
                print(f"[OPTIMIZED] Using consistent mapping: {original_speaker} -> {mapped_speaker}")
                return mapped_speaker
            else:
                print(f"[OPTIMIZED] Inconsistent mapping detected, reassigning {original_speaker}")
                # Remove the inconsistent mapping
                del self.diarization_speaker_map[original_speaker]
        
        # Find the most similar existing speaker
        similar_speaker = self.find_similar_speaker_in_session(embedding)
        if similar_speaker:
            # Create consistent mapping
            self.diarization_speaker_map[original_speaker] = similar_speaker
            print(f"[OPTIMIZED] Created consistent mapping: {original_speaker} -> {similar_speaker}")
            return similar_speaker
        
        # Create new speaker if no good match found
        speaker_id = f"SPEAKER_{self.speaker_counter:02d}"
        self.speaker_counter += 1
        self.diarization_speaker_map[original_speaker] = speaker_id
        print(f"[OPTIMIZED] Created new speaker: {original_speaker} -> {speaker_id}")
        return speaker_id
    
    def get_all_speakers(self) -> Dict:
        """Get all speakers from cache."""
        with self.lock:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = conn.cursor()
            
            cursor.execute("SELECT speaker_id, name, first_seen, last_seen, segment_count FROM speakers")
            results = cursor.fetchall()
            
            speakers = {}
            for row in results:
                speakers[row[0]] = {
                    'name': row[1],
                    'first_seen': row[2],
                    'last_seen': row[3],
                    'segment_count': row[4]
                }
            
            conn.close()
            return speakers
    
    def clear_all_speakers(self):
        """Clear all speakers from database and cache."""
        with self.lock:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM speakers")
            conn.commit()
            conn.close()
            
            self.cache.clear()
            print("[DATABASE] Speaker database and cache cleared")


class OptimizedSpeakerRecognition:
    """Optimized speaker recognition system with SpeechBrain and performance improvements."""
    
    def __init__(self, hf_token: str):
        self.db = OptimizedSpeakerDatabase()
        self.speaker_counter = 0
        self.hf_token = hf_token
        
        # Persistent mapping between diarization speakers and our speaker IDs
        self.diarization_speaker_map = {}
        
        # Cross-segment speaker memory for entire session
        # Simple speaker recognition for whole audio processing
        self.similarity_threshold = 0.6  # Threshold for speaker similarity
        
        # Initialize SpeechBrain speaker encoder (ECAPA-TDNN - state-of-the-art)
        print("[SPEECHBRAIN] Loading ECAPA-TDNN speaker encoder...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Use model directly to avoid Windows symlink issues
        print("[SPEECHBRAIN] Loading model directly from HuggingFace...")
        self.speaker_encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device}
        )
        
        # Performance optimizations for SpeechBrain
        self.speaker_encoder.eval()  # Set to evaluation mode
        if device == "cuda":
            self.speaker_encoder = self.speaker_encoder.half()  # Use half precision for speed
        
        # Cache for embeddings to avoid recomputation
        self.embedding_cache = {}
        self.cache_max_size = 1000  # Limit cache size
        
        print(f"[SPEECHBRAIN] Speaker encoder loaded successfully on {device}")
        print(f"[SPEECHBRAIN] Using half precision: {device == 'cuda'}")
        
        # Initialize optimized diarization pipeline
        from optimized_diarization import OptimizedDiarizationPipeline
        self.pipeline = OptimizedDiarizationPipeline(hf_token)
        
        print("[OPTIMIZED] Speaker recognition system initialized with SpeechBrain")
        print(f"[OPTIMIZED] Known speakers: {len(self.db.get_all_speakers())}")
    
    def reset_session(self):
        """Reset the session - clear database and reset speaker counter."""
        self.db.clear_all_speakers()
        self.speaker_counter = 0
        self.diarization_speaker_map = {}  # Clear persistent mapping
        self.embedding_cache.clear()  # Clear embedding cache
        print("[OPTIMIZED] Session reset - speaker database, cache, and counter reset")
    
    def extract_speaker_embedding(self, audio_array: np.ndarray, start_time: float, end_time: float, sample_rate: int = 16000) -> np.ndarray:
        """Extract speaker embedding using SpeechBrain ECAPA-TDNN (high accuracy with caching)."""
        # Convert time to samples
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        segment = audio_array[start_sample:end_sample]
        
        print(f"[OPTIMIZED] Extracting embedding: {start_time:.2f}s-{end_time:.2f}s, {len(segment)} samples")
        
        # Create cache key based on audio content hash
        segment_hash = hash(segment.tobytes())
        if segment_hash in self.embedding_cache:
            print(f"[OPTIMIZED] Using cached embedding")
            return self.embedding_cache[segment_hash]
        
        # Ensure minimum length for SpeechBrain (at least 0.5 seconds)
        min_length = int(0.5 * sample_rate)
        if len(segment) < min_length:
            # Pad with zeros if too short
            padding = np.zeros(min_length - len(segment), dtype=np.float32)
            segment = np.concatenate([segment, padding])
        
        # Convert to torch tensor and ensure correct format for SpeechBrain
        audio_tensor = torch.tensor(segment, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        
        # Extract speaker embedding using SpeechBrain ECAPA-TDNN
        with torch.no_grad():
            embedding = self.speaker_encoder.encode_batch(audio_tensor)
            # Convert to numpy and flatten
            embedding = embedding.squeeze().cpu().numpy()
        
        print(f"[OPTIMIZED] Generated embedding: shape={embedding.shape}, mean={np.mean(embedding):.3f}, std={np.std(embedding):.3f}")
        
        # Cache the embedding (with size limit)
        if len(self.embedding_cache) < self.cache_max_size:
            self.embedding_cache[segment_hash] = embedding
        
        return embedding
    
    def process_audio_array(self, audio_array: np.ndarray, sample_rate: int = 16000) -> Dict:
        """Process audio array with optimized speaker recognition."""
        print(f"\n[OPTIMIZED] ===== STARTING OPTIMIZED SPEAKER RECOGNITION =====")
        print(f"[OPTIMIZED] Processing audio array: {len(audio_array)} samples, {sample_rate} Hz")
        
        # Use optimized diarization pipeline
        result = self.pipeline.process_audio_array(audio_array, sample_rate)
        segments = result.get('segments', [])
        
        print(f"[OPTIMIZED] Diarization returned {len(segments)} segments")
        
        # Process each segment with simple speaker recognition
        processed_segments = []
        # Simple speaker recognition within the audio file
        for i, segment in enumerate(segments):
            start_time = segment.get('start', 0)
            end_time = segment.get('end', 0)
            text = segment.get('transcription', '')
            original_speaker = segment.get('speaker_id', 'unknown')
            
            # Use diarization speaker directly (no cross-segment memory needed)
            speaker_id = original_speaker
            print(f"[OPTIMIZED] Segment {i+1}: Using diarization speaker {speaker_id}")
            
            # Update the segment with speaker ID
            processed_segments.append({
                'start': start_time,
                'end': end_time,
                'speaker_id': speaker_id,
                'transcription': text,
                'confidence': segment.get('confidence', 1.0),
                'duration': end_time - start_time
            })
        
        # Update result with processed segments
        result['segments'] = processed_segments
        result['speakers'] = list(set(seg['speaker_id'] for seg in processed_segments))
        result['processing_method'] = 'speaker_recognition_speechbrain'
        
        print(f"[OPTIMIZED] ===== OPTIMIZED SPEAKER RECOGNITION COMPLETE =====")
        print(f"[OPTIMIZED] Final result: {len(processed_segments)} segments, {len(result['speakers'])} speakers")
        
        return result
    
    def get_all_speakers(self) -> Dict:
        """Get information about all known speakers."""
        return self.db.get_all_speakers()


def main():
    """Main function to test optimized speaker recognition."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimized Speaker Recognition")
    parser.add_argument("--audio", type=str, help="Audio file to process")
    parser.add_argument("--hf_token", type=str, help="Hugging Face token")
    parser.add_argument("--reset", action="store_true", help="Reset speaker database")
    args = parser.parse_args()
    
    if not args.hf_token:
        args.hf_token = os.getenv("HF_TOKEN")
        if not args.hf_token:
            print("ERROR: Please provide Hugging Face token")
            return
    
    # Initialize optimized system
    system = OptimizedSpeakerRecognition(args.hf_token)
    
    if args.reset:
        system.reset_session()
        return
    
    if args.audio:
        # Process single audio file
        import soundfile as sf
        audio_array, sample_rate = sf.read(args.audio)
        result = system.process_audio_array(audio_array, sample_rate)
        
        print("\n" + "="*60)
        print("OPTIMIZED SPEAKER RECOGNITION RESULTS")
        print("="*60)
        
        for i, segment in enumerate(result['segments']):
            print(f"Segment {i+1}: {segment['speaker_id']} ({segment['start']:.2f}s - {segment['end']:.2f}s)")
            if segment['transcription'].strip():
                print(f"  Text: '{segment['transcription']}'")
            print()
        
        print(f"Total speakers in database: {len(result['speakers'])}")
    
    else:
        # Show current speaker database
        speakers = system.get_all_speakers()
        print(f"\nCurrent speakers in database: {len(speakers)}")
        for speaker_id, data in speakers.items():
            print(f"  {speaker_id}: {data['name']} (seen {data['segment_count']} times)")


if __name__ == "__main__":
    main()
