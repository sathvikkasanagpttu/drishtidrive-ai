/**
 * In-browser audio player for DrishtiDrive AI alert subsystem.
 *
 * Provides dual-layer audio playback:
 * 1. Primary: Plays WAV alert tones and prerecorded MP3 voice directives from /api/audio.
 * 2. Fallback: Web Audio API oscillator synthesis + Web Speech API synthesis for offline resilience.
 */

let activeToneAudio = null;
let activeVoiceAudio = null;
let audioCtx = null;

const STAGE_VOICE_MESSAGES = {
  1: 'Please stay alert. Keep your eyes on the road.',
  2: 'Warning! You may be getting drowsy. Please find a safe place to stop.',
  3: 'Danger! Pull over safely as soon as possible. Do not continue driving while drowsy.',
};

/**
 * Stop any ongoing browser alert audio immediately.
 */
export function stopBrowserAlert() {
  if (activeToneAudio) {
    try {
      activeToneAudio.pause();
      activeToneAudio.currentTime = 0;
    } catch (e) {}
    activeToneAudio = null;
  }
  if (activeVoiceAudio) {
    try {
      activeVoiceAudio.pause();
      activeVoiceAudio.currentTime = 0;
    } catch (e) {}
    activeVoiceAudio = null;
  }
  if (typeof window !== 'undefined' && window.speechSynthesis) {
    try {
      window.speechSynthesis.cancel();
    } catch (e) {}
  }
}

/**
 * Synthesize alert tone using Web Audio API oscillator fallback.
 */
function playOscillatorTone(stage, volume = 0.8) {
  try {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextClass) return;
    if (!audioCtx) audioCtx = new AudioContextClass();
    if (audioCtx.state === 'suspended') {
      audioCtx.resume();
    }

    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    const now = audioCtx.currentTime;

    osc.type = stage === 3 ? 'sawtooth' : 'sine';
    const vol = Math.max(0.01, Math.min(1.0, volume));
    gain.gain.setValueAtTime(vol * 0.3, now);

    if (stage === 1) {
      // 520Hz gentle beep for 0.35s
      osc.frequency.setValueAtTime(520, now);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.35);
      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.start(now);
      osc.stop(now + 0.36);
    } else if (stage === 2) {
      // 750Hz dual pulse
      osc.frequency.setValueAtTime(750, now);
      gain.gain.setValueAtTime(vol * 0.4, now);
      gain.gain.setValueAtTime(0.001, now + 0.18);
      gain.gain.setValueAtTime(vol * 0.4, now + 0.28);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.5);
      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.start(now);
      osc.stop(now + 0.52);
    } else {
      // Stage 3 piercing alternating siren 880Hz <-> 1100Hz
      osc.frequency.setValueAtTime(880, now);
      osc.frequency.linearRampToValueAtTime(1100, now + 0.25);
      osc.frequency.linearRampToValueAtTime(880, now + 0.5);
      osc.frequency.linearRampToValueAtTime(1100, now + 0.75);
      osc.frequency.linearRampToValueAtTime(880, now + 1.0);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 1.2);
      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.start(now);
      osc.stop(now + 1.25);
    }
  } catch (err) {
    console.warn('[DrishtiDrive Audio] Web Audio oscillator fallback error:', err);
  }
}

/**
 * Fallback voice directive using browser Web Speech API.
 */
function playWebSpeech(stage, volume = 0.8) {
  if (typeof window === 'undefined' || !window.speechSynthesis) return;
  try {
    const text = STAGE_VOICE_MESSAGES[stage] || STAGE_VOICE_MESSAGES[1];
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.volume = Math.max(0.1, Math.min(1.0, volume));
    utterance.rate = 1.0;
    utterance.pitch = stage === 3 ? 1.2 : 1.0;
    window.speechSynthesis.speak(utterance);
  } catch (e) {
    console.warn('[DrishtiDrive Audio] Web Speech fallback error:', e);
  }
}

/**
 * Play stage alert audio in browser: sequential WAV tone followed by MP3 voice.
 * Automatically falls back to Web Audio oscillator and Web Speech API.
 *
 * @param {number} stage - Alert stage (1, 2, or 3)
 * @param {number} volume - Volume (0.0 to 1.0)
 * @returns {Promise<boolean>}
 */
export async function playBrowserAlert(stage = 1, volume = 0.8) {
  const targetStage = [1, 2, 3].includes(Number(stage)) ? Number(stage) : 1;
  const vol = Math.max(0.01, Math.min(1.0, volume));

  stopBrowserAlert();

  const toneUrl = `/api/audio/tone/${targetStage}?t=${Date.now()}`;
  const voiceUrl = `/api/audio/voice/${targetStage}?t=${Date.now()}`;

  try {
    const toneAudio = new Audio(toneUrl);
    toneAudio.volume = vol;
    activeToneAudio = toneAudio;

    const playVoice = () => {
      try {
        const voiceAudio = new Audio(voiceUrl);
        voiceAudio.volume = vol;
        activeVoiceAudio = voiceAudio;
        voiceAudio.play().catch((err) => {
          console.warn('[DrishtiDrive Audio] Voice MP3 failed, using Web Speech:', err);
          playWebSpeech(targetStage, vol);
        });
      } catch (err) {
        playWebSpeech(targetStage, vol);
      }
    };

    toneAudio.onended = () => {
      activeToneAudio = null;
      playVoice();
    };

    toneAudio.onerror = () => {
      console.warn('[DrishtiDrive Audio] Tone WAV load error, using oscillator fallback');
      playOscillatorTone(targetStage, vol);
      setTimeout(() => playVoice(), 400);
    };

    await toneAudio.play();
    return true;
  } catch (err) {
    console.warn('[DrishtiDrive Audio] Primary audio playback blocked/failed, using fallback synth:', err);
    playOscillatorTone(targetStage, vol);
    setTimeout(() => playWebSpeech(targetStage, vol), 400);
    return false;
  }
}
