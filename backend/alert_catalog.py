"""Multi-language alert message catalog and localized voice announcements.

Provides standardized, firm, and understandable messages across alert stages.
Default language is English, with verified Hindi and Marathi localizations.

Safety principles:
- Short, clear, and firm without causing unnecessary panic or abrupt braking.
- Instructs the driver to stop safely when drowsy.
- Never claims the system guarantees accident prevention.
"""
from __future__ import annotations

from typing import Dict, List, Optional
from backend.temporal_state import AlertStage


# Standard alert texts across stages and supported languages
ALERT_MESSAGES: Dict[str, Dict[AlertStage, str]] = {
    "en": {
        AlertStage.STAGE_0_NORMAL: "Normal monitoring active. Stay attentive.",
        AlertStage.STAGE_1_EARLY: "Please stay alert. Keep your eyes on the road.",
        AlertStage.STAGE_2_DROWSY: "Warning! You may be getting drowsy. Please find a safe place to stop.",
        AlertStage.STAGE_3_CRITICAL: "Danger! Pull over safely as soon as possible. Do not continue driving while drowsy.",
    },
    "hi": {
        AlertStage.STAGE_0_NORMAL: "निगरानी सक्रिय है। सतर्क रहें।",
        AlertStage.STAGE_1_EARLY: "कृपया सतर्क रहें। अपनी आँखें सड़क पर रखें।",
        AlertStage.STAGE_2_DROWSY: "चेतावनी! आपको नींद आ रही हो सकती है। कृपया रुकने के लिए सुरक्षित स्थान खोजें।",
        AlertStage.STAGE_3_CRITICAL: "खतरा! यथाशीघ्र सुरक्षित रूप से गाड़ी रोकें। झपकी की स्थिति में गाड़ी न चलाएं।",
    },
    "mr": {
        AlertStage.STAGE_0_NORMAL: "निरीक्षण सुरू आहे. लक्ष ठेवा.",
        AlertStage.STAGE_1_EARLY: "कृपया सावध रहा. आपले लक्ष रस्त्यावर ठेवा.",
        AlertStage.STAGE_2_DROWSY: "इशारा! तुम्हाला झोप येत असावी. कृपया थांबण्यासाठी सुरक्षित जागा शोधा.",
        AlertStage.STAGE_3_CRITICAL: "धोका! लवकरात लवकर सुरक्षितपणे गाडी बाजूला घ्या. झोपेत असताना गाडी चालवू नका.",
    },
    "te": {
        AlertStage.STAGE_0_NORMAL: "సాధారణ పర్యవేక్షణ చురుకుగా ఉంది. శ్రద్ధగా ఉండండి.",
        AlertStage.STAGE_1_EARLY: "దయచేసి అప్రమత్తంగా ఉండండి. మీ దృష్టిని రోడ్డుపై ఉంచండి.",
        AlertStage.STAGE_2_DROWSY: "హెచ్చరిక! మీకు నిద్ర వస్తుండవచ్చు. దయచేసి సురక్షితమైన ప్రదేశంలో ఆగండి.",
        AlertStage.STAGE_3_CRITICAL: "ప్రమాదం! వీలైనంత త్వరగా సురక్షితంగా వాహనం నిలపండి. నిద్రమత్తులో వాహనం నడపవద్దు.",
    },
    "ta": {
        AlertStage.STAGE_0_NORMAL: "கண்காணிப்பு செயலில் உள்ளது. விழிப்புடன் இருங்கள்.",
        AlertStage.STAGE_1_EARLY: "தயவுசெய்து விழிப்புடன் இருங்கள். உங்கள் பார்வையை சாலையில் வைக்கவும்.",
        AlertStage.STAGE_2_DROWSY: "எச்சரிக்கை! உங்களுக்கு தூக்கம் வரக்கூடும். பாதுகாப்பான இடத்தில் வாகனத்தை நிறுத்துங்கள்.",
        AlertStage.STAGE_3_CRITICAL: "ஆபத்து! உடனடியாக வாகனத்தை பாதுகாப்பாக நிறுத்துங்கள். தூக்க கலக்கத்தில் வாகனத்தை ஓட்டாதீர்கள்.",
    },
    "kn": {
        AlertStage.STAGE_0_NORMAL: "ಮೇಲ್ವಿಚಾರಣೆ ಸಕ್ರಿಯವಾಗಿದೆ. ಎಚ್ಚರಿಕೆಯಿಂದಿರಿ.",
        AlertStage.STAGE_1_EARLY: "ದಯವಿಟ್ಟು ಜಾಗರೂಕರಾಗಿರಿ. ನಿಮ್ಮ ಗಮನ ರಸ್ತೆಯ ಮೇಲಿರಲಿ.",
        AlertStage.STAGE_2_DROWSY: "ಎಚ್ಚರಿಕೆ! ನಿಮಗೆ ನಿದ್ರೆ ಬರುತ್ತಿರಬಹುದು. ದಯವಿಟ್ಟು ಸುರಕ್ಷಿತ ಸ್ಥಳದಲ್ಲಿ ನಿಲ್ಲಿಸಿ.",
        AlertStage.STAGE_3_CRITICAL: "ಅಪಾಯ! ತಕ್ಷಣ ವಾಹನವನ್ನು ಸುರಕ್ಷಿತವಾಗಿ ಪಕ್ಕಕ್ಕೆ ನಿಲ್ಲಿಸಿ. ನಿದ್ರೆಯ ಮಂಪರಿನಲ್ಲಿ ಚಾಲನೆ ಮಾಡಬೇಡಿ.",
    },
    "bn": {
        AlertStage.STAGE_0_NORMAL: "নজরদারি সক্রিয় রয়েছে। সতর্ক থাকুন।",
        AlertStage.STAGE_1_EARLY: "দয়া করে সতর্ক থাকুন। রাস্তায় চোখ রাখুন।",
        AlertStage.STAGE_2_DROWSY: "সতর্কতা! আপনার ঘুম আসতে পারে। দয়া করে নিরাপদ স্থানে থামুন।",
        AlertStage.STAGE_3_CRITICAL: "বিপদ! অবিলম্বে নিরাপদে গাড়ি থামান। তন্দ্রাচ্ছন্ন অবস্থায় গাড়ি চালাবেন না।",
    },
}

ALERT_TITLES: Dict[str, Dict[AlertStage, str]] = {
    "en": {
        AlertStage.STAGE_0_NORMAL: "ATTENTIVE",
        AlertStage.STAGE_1_EARLY: "STAGE 1 - ATTENTION WARNING",
        AlertStage.STAGE_2_DROWSY: "STAGE 2 - DROWSINESS WARNING",
        AlertStage.STAGE_3_CRITICAL: "STAGE 3 - CRITICAL WARNING",
    },
    "hi": {
        AlertStage.STAGE_0_NORMAL: "सतर्क",
        AlertStage.STAGE_1_EARLY: "चरण १ - ध्यान दें",
        AlertStage.STAGE_2_DROWSY: "चरण २ - सुस्ती चेतावनी",
        AlertStage.STAGE_3_CRITICAL: "चरण ३ - गंभीर सुरक्षा चेतावनी",
    },
    "mr": {
        AlertStage.STAGE_0_NORMAL: "सावध",
        AlertStage.STAGE_1_EARLY: "टप्पा १ - लक्ष द्या",
        AlertStage.STAGE_2_DROWSY: "टप्पा २ - डुलकी इशारा",
        AlertStage.STAGE_3_CRITICAL: "टप्पा ३ - गंभीर सुरक्षा इशारा",
    },
    "te": {
        AlertStage.STAGE_0_NORMAL: "సాధారణం",
        AlertStage.STAGE_1_EARLY: "దశ 1 - శ్రద్ధ హెచ్చరిక",
        AlertStage.STAGE_2_DROWSY: "దశ 2 - నిద్రమత్తు హెచ్చరిక",
        AlertStage.STAGE_3_CRITICAL: "దశ 3 - తీవ్రమైన హెచ్చరిక",
    },
    "ta": {
        AlertStage.STAGE_0_NORMAL: "கவனம்",
        AlertStage.STAGE_1_EARLY: "நிலை 1 - கவன எச்சரிக்கை",
        AlertStage.STAGE_2_DROWSY: "நிலை 2 - தூக்க எச்சரிக்கை",
        AlertStage.STAGE_3_CRITICAL: "நிலை 3 - தீவிர ஆபத்து எச்சரிக்கை",
    },
    "kn": {
        AlertStage.STAGE_0_NORMAL: "ಎಚ್ಚರ",
        AlertStage.STAGE_1_EARLY: "ಹಂತ 1 - ಗಮನ ಎಚ್ಚರಿಕೆ",
        AlertStage.STAGE_2_DROWSY: "ಹಂತ 2 - ನಿದ್ರೆ ಎಚ್ಚರಿಕೆ",
        AlertStage.STAGE_3_CRITICAL: "ಹಂತ 3 - ಗಂಭೀರ ಅಪಾಯ ಎಚ್ಚರಿಕೆ",
    },
    "bn": {
        AlertStage.STAGE_0_NORMAL: "সতর্ক",
        AlertStage.STAGE_1_EARLY: "পর্যায় ১ - মনোযোগ সতর্কতা",
        AlertStage.STAGE_2_DROWSY: "পর্যায় ২ - তন্দ্রা সতর্কতা",
        AlertStage.STAGE_3_CRITICAL: "পর্যায় ৩ - চরম বিপদ সতর্কতা",
    },
}

DISTRACTION_MESSAGES: Dict[str, Dict[str, str]] = {
    "en": {
        "HEAD_POSE_AWAY": "Focus on the road! Looking away from the road detected.",
        "PHONE_DISTRACTION": "Put your phone down! Mobile phone use while driving is dangerous.",
        "NO_SEATBELT": "Please fasten your seatbelt for safety.",
    },
    "hi": {
        "HEAD_POSE_AWAY": "सड़क पर ध्यान दें! सड़क से ध्यान हटने का पता चला है।",
        "PHONE_DISTRACTION": "कृपया फोन नीचे रखें! गाड़ी चलाते समय फोन का उपयोग खतरनाक है।",
        "NO_SEATBELT": "कृपया सुरक्षा के लिए अपनी सीटबेल्ट बांधें।",
    },
    "mr": {
        "HEAD_POSE_AWAY": "रस्त्याकडे लक्ष द्या! तुमचे लक्ष रस्त्यावरून विचलित झाले आहे.",
        "PHONE_DISTRACTION": "कृपया फोन बाजूला ठेवा! वाहन चालवताना फोन वापरणे धोकादायक आहे.",
        "NO_SEATBELT": "कृपया सुरक्षिततेसाठी सीटबेल्ट लावा.",
    },
    "te": {
        "HEAD_POSE_AWAY": "రోడ్డుపై దృష్టి పెట్టండి! రోడ్డు నుండి దృష్టి మళ్ళించబడింది.",
        "PHONE_DISTRACTION": "దయచేసి ఫోన్ పక్కన పెట్టండి! డ్రైవింగ్ చేస్తూ ఫోన్ ఉపయోగించడం ప్రమాదకరం.",
        "NO_SEATBELT": "దయచేసి భద్రత కోసం సీట్‌బెల్ట్ ధరించండి.",
    },
    "ta": {
        "HEAD_POSE_AWAY": "சாலையில் கவனம் செலுத்துங்கள்! சாலை கவனம் தவறியது கண்டறியப்பட்டது.",
        "PHONE_DISTRACTION": "போனை கீழே வையுங்கள்! வாகனம் ஓட்டும் போது போன் பேசுவது ஆபத்தானது.",
        "NO_SEATBELT": "பாதுகாப்பிற்காக சீட்பெல்ட் அணியவும்.",
    },
    "kn": {
        "HEAD_POSE_AWAY": "ರಸ್ತೆಯ ಮೇಲೆ ಗಮನವಿರಲಿ! ರಸ್ತೆಯಿಂದ ಗಮನ ಬೇರೆಡೆ ತಿರುಗಿದೆ.",
        "PHONE_DISTRACTION": "ಫೋನ್ ಕೆಳಗಿಡಿ! ಚಾಲನೆ ಮಾಡುವಾಗ ಮೊಬೈಲ್ ಬಳಕೆ ಅಪಾಯಕಾರಿ.",
        "NO_SEATBELT": "ಸುರಕ್ಷತೆಗಾಗಿ ಸೀಟ್‌ಬೆಲ್ಟ್ ಧರಿಸಿ.",
    },
    "bn": {
        "HEAD_POSE_AWAY": "রাস্তায় মনোযোগ দিন! রাস্তা থেকে নজর সরানোর প্রমাণ পাওয়া গেছে।",
        "PHONE_DISTRACTION": "ফোনটি নামিয়ে রাখুন! গাড়ি চালানোর সময় ফোন ব্যবহার বিপজ্জনক।",
        "NO_SEATBELT": "নিরাপত্তার জন্য সিটবেল্ট বাঁধুন।",
    },
}

# Offline Piper TTS voice catalog mapping
PIPER_VOICES: Dict[str, str] = {
    "en": "en_US-lessac-medium",
    "hi": "hi_IN-hindi-medium",
    "mr": "mr_IN-marathi-medium",
    "te": "te_IN-telugu-medium",
    "ta": "ta_IN-tamil-medium",
    "kn": "kn_IN-kannada-medium",
    "bn": "bn_IN-bengali-medium",
}


class AlertMessageCatalog:
    """Catalog manager for multi-language alert text and voice prompts."""

    DEFAULT_LANGUAGE = "en"
    SUPPORTED_LANGUAGES: List[str] = ["en", "hi", "mr", "te", "ta", "kn", "bn"]

    @classmethod
    def get_message(cls, stage: AlertStage, language: str = DEFAULT_LANGUAGE) -> str:
        """Retrieve localized spoken message for a given alert stage."""
        lang = language.lower() if language.lower() in ALERT_MESSAGES else cls.DEFAULT_LANGUAGE
        return ALERT_MESSAGES[lang].get(
            stage,
            ALERT_MESSAGES[cls.DEFAULT_LANGUAGE][stage],
        )

    @classmethod
    def get_title(cls, stage: AlertStage, language: str = DEFAULT_LANGUAGE) -> str:
        """Retrieve localized display title for a given alert stage."""
        lang = language.lower() if language.lower() in ALERT_TITLES else cls.DEFAULT_LANGUAGE
        return ALERT_TITLES[lang].get(
            stage,
            ALERT_TITLES[cls.DEFAULT_LANGUAGE][stage],
        )

    @classmethod
    def get_distraction_message(cls, distraction_type: str, language: str = DEFAULT_LANGUAGE) -> str:
        """Retrieve localized distraction warning message."""
        lang = language.lower() if language.lower() in DISTRACTION_MESSAGES else cls.DEFAULT_LANGUAGE
        return DISTRACTION_MESSAGES[lang].get(
            distraction_type,
            DISTRACTION_MESSAGES[cls.DEFAULT_LANGUAGE].get(distraction_type, "Distraction detected. Stay attentive."),
        )

    @classmethod
    def get_piper_voice(cls, language: str = DEFAULT_LANGUAGE) -> str:
        """Retrieve Piper TTS voice model name for language."""
        lang = language.lower() if language.lower() in PIPER_VOICES else cls.DEFAULT_LANGUAGE
        return PIPER_VOICES[lang]

    @classmethod
    def is_language_supported(cls, language: str) -> bool:
        """Check if requested language code is supported."""
        return language.lower() in cls.SUPPORTED_LANGUAGES

