import re

class GeezNormalizer:
    def __init__(self):
        # የሆሄያት መቀያየሪያ ካርታ (Homophones mapping)
        # በተጠቃሚው ፍላጎት መሰረት የተዘጋጀ
        self.char_map = {
            "ሀ": "ሃ", "ሐ": "ሃ", "ሓ": "ሃ", "ኅ": "ሃ", "ኻ": "ሃ", "ኃ": "ሃ",
            "ዅ": "ሁ", "ሗ": "ኋ", "ኁ": "ሁ", "ኂ": "ሂ", "ኄ": "ሄ", "ዄ": "ሄ",
            "ኅ": "ህ", "ኆ": "ሆ", "ሑ": "ሁ", "ሒ": "ሂ", "ሔ": "ሄ", "ሕ": "ህ",
            "ሖ": "ሆ", "ኾ": "ሆ", "ሠ": "ሰ", "ሡ": "ሱ", "ሢ": "ሲ", "ሣ": "ሳ",
            "ሤ": "ሴ", "ሥ": "ስ", "ሦ": "ሶ", "ዐ": "አ", "ዑ": "ኡ", "ዒ": "ኢ",
            "ዓ": "አ", "ኣ": "አ", "ዔ": "ኤ", "ዕ": "እ", "ዖ": "ኦ", "ፀ": "ጸ",
            "ፁ": "ጹ", "ጺ": "ፂ", "ጻ": "ፃ", "ጼ": "ፄ", "ፅ": "ጽ", "ፆ": "ጾ",
            "ሼ": "ሸ", "ሺ": "ሽ", "ዲ": "ድ", "ጄ": "ጀ", "ጂ": "ጅ", "ዉ": "ው",
            "ዎ": "ወ", "ዴ": "ደ", "ቼ": "ቸ", "ቺ": "ች", "ዬ": "የ", "ዪ": "ይ",
            "ጬ": "ጨ", "ጪ": "ጭ", "ኜ": "ኘ", "ኚ": "ኝ", "ዤ": "ዠ", "ዢ": "ዥ",
            "ቊ": "ቁ", "ኵ": "ኩ"
        }
        
        # characters to be removed
        # የሚወገዱ የአማርኛ ስርዓተ ነጥቦች
        self.punctuation_pattern = r'[፠፡።፣፤፥፦፧፨]'
        
        # የሳልሳዊ እና ሳብአዊ ሆሄያት ውህደት (Labialized characters)
        self.labialized_map = [
            (r'(ሉ[ዋአ])', 'ሏ'), (r'(ሙ[ዋአ])', 'ሟ'), (r'(ቱ[ዋአ])', 'ቷ'),
            (r'(ሩ[ዋአ])', 'ሯ'), (r'(ሱ[ዋአ])', 'ሷ'), (r'(ሹ[ዋአ])', 'ሿ'),
            (r'(ቁ[ዋአ])', 'ቋ'), (r'(ቡ[ዋአ])', 'ቧ'), (r'(ቹ[ዋአ])', 'ቿ'),
            (r'(ሁ[ዋአ])', 'ኋ'), (r'(ኑ[ዋአ])', 'ኗ'), (r'(ኙ[ዋአ])', 'ኟ'),
            (r'(ኩ[ዋአ])', 'ኳ'), (r'(ዙ[ዋአ])', 'ዟ'), (r'(ጉ[ዋአ])', 'ጓ'),
            (r'(ደ[ዋአ])', 'ዷ'), (r'(ጡ[ዋአ])', 'ጧ'), (r'(ጩ[ዋአ])', 'ጯ'),
            (r'(ጹ[ዋአ])', 'ጿ'), (r'(ፉ[ዋአ])', 'ፏ')
        ]

    def normalize(self, text, remove_punctuation=True):
        if not text:
            return ""

        # remove punctuations 
        # 1. ስርዓተ ነጥቦችን ማስወገድ
        if remove_punctuation:
            text = re.sub(self.punctuation_pattern, ' ', text)
        
        # replace characters 
        # 2. ሆሄያትን መተካት (የአንድ ለአንድ መተካት)
        # str.translate ለትልቅ ጽሁፍ እጅግ ፈጣን ነው
        trans_table = str.maketrans(self.char_map)
        text = text.translate(trans_table)
        
        # 3. የሳልሳዊ እና ሳብአዊ ውህደቶችን ማስተካከል (Regex)
        for pattern, replacement in self.labialized_map:
            text = re.sub(pattern, replacement, text)
            
        # remove spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text


if __name__ == "__main__":
    normalizer = GeezNormalizer()
    sample_text = "ኃይለ ሥላሴ ሐመረ ኖኅ፡ በዐፄ ቴዎድሮስ ጊዜ። ሉአላዊነት ፠"
    
    print(f"Original: {sample_text}")
    
    clean_text = normalizer.normalize(sample_text, remove_punctuation=True)
    print(f"Normalized (remove_punctuation=True): {clean_text}")
    
    clean_text_no_remove = normalizer.normalize(sample_text, remove_punctuation=False)
    print(f"Normalized (remove_punctuation=False): {clean_text_no_remove}")
